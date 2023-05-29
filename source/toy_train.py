from __future__ import annotations
import torch
import torch.utils.data as data
from PIL import Image
import os.path as osp
import numpy as np
torch.manual_seed(0)
import tifffile
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .model import UNet
from .model_old import UNet_old
from .sliced_wasserstein import sliced_wasserstein_distance
from .train import make_model
from .train import prepare_labels, save
from .utils import merge_canal
from statistics import NormalDist
import wandb
import pandas
wandb.login()

class LinearNormAct(torch.nn.Sequential):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn: bool = False,
        act: bool = True
    ) -> None:
        super().__init__()
        self.add_module('linear', torch.nn.Linear(in_features, out_features, bias=not bn))
        self.add_module('norm', torch.nn.BatchNorm2d(out_features) if bn else torch.nn.Identity())
        self.add_module('act', torch.nn.ReLU(inplace=True) if act else torch.nn.Identity())


class MLP(torch.nn.Module):

    def __init__(
        self,  widths: tuple[int] = (16, 32), bn: bool = False,
    ) -> None:
        super().__init__()
        layers = list()
        for i, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            act = i != len(widths) - 2  # activation for all layers except the last
            layers.append(LinearNormAct(in_features, out_features, bn=bn, act=act))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AE(torch.nn.Module):

    def __init__(self, image_size: int, widths: tuple[int], bn: bool = False) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.encoder = MLP(widths=(image_size ** 2, *widths), bn=bn)
        self.decoder = MLP(widths=(*widths[::-1], image_size ** 2), bn=bn)
        self.unflatten = torch.nn.Unflatten(1, (image_size, image_size))
        self._latent_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.flatten(x)).relu()
        x_hat = self.unflatten(self.decoder(z))
        return x_hat

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def num_params(self) -> dict[str, int]:
        return dict(encoder=self.encoder.num_params, decoder=self.decoder.num_params)

    
class SegmentationDataSet(data.Dataset):
    
    def __init__(self, root, list_path):
            self.root = root
            self.list_path = list_path
            self.list_ids = [i_id.strip() for i_id in open(list_path)]

    def __len__(self):
        return len(self.list_ids)


    def __getitem__(self, index: int):
        name = self.list_ids[index]
        img = Image.open(osp.join(self.root, "img/%s" % (name))).convert("RGB")
        #print(np.shape(img))
        label = Image.open(osp.join(self.root, "label/%s" % (name))).convert("RGB")
        img_np = np.asarray(img)
        label_np = np.asarray(label)        
        img_np = img_np.transpose((2,0,1))       #Channel Arrangement
        label_np = label_np.transpose((2,0,1))
        img_np = img_np/255                      #NORMALIZATION
        label_np = label_np/255                 #Should we normalize the mask ?                
        return {
            'image': torch.as_tensor(img_np.copy()).float().contiguous(),
            'mask': torch.as_tensor(label_np.copy()).float().contiguous()
        }
    
def import_data(root, list_path, batch_size, taille=1) :
    ds = SegmentationDataSet(root=root, list_path=list_path)    
    train, test = torch.utils.data.random_split(ds,[0.8, 0.2])
    
    if taille == 1 :
        data_train = torch.utils.data.DataLoader(train, 
                                   batch_size=batch_size,
                                   shuffle=True)   
        data_test = torch.utils.data.DataLoader(test, 
                                   batch_size=batch_size,
                                   shuffle=True)   
    
    else :
        reduce_size = taille
        inv_reduce_size = 1-reduce_size
        train_reduce, _ = torch.utils.data.random_split(train, [reduce_size, inv_reduce_size])
        data_train = torch.utils.data.DataLoader(train_reduce, 
                                   batch_size=batch_size,
                                   shuffle=True)   
        data_test = torch.utils.data.DataLoader(test, 
                                   batch_size=batch_size,
                                   shuffle=True)   
    return data_train, data_test
    
def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6
) -> Tensor:
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
) -> Tensor:
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon
        )
    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DiceLoss(torch.nn.Module):
    def __init__(self, multiclass: bool = True) -> None:
        super().__init__()
        self.multiclass = multiclass
        
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return dice_loss(outputs, targets, multiclass=self.multiclass)


def train(dataset, n_epoch, model, criterion, optimizer, grad_scaler, device, saving_root) :
    print("Training")
    
    wandb.init(
    # Set the project where this run will be logged
    project="ICCV_2023", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{saving_root}", 
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "U_Net",
    "dataset": saving_root,
    "epochs": n_epoch,
    })
    
    for epoch in range(n_epoch) :
        running_loss = 0.0
        for batch in dataset : 
            model.train()
            images = batch['image']
            masks = batch['mask']
            assert images.shape[1] == model.n_channels
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=False):
                pred = model(images)
            ############################
            # (2) Update model network: minimize Lseg(Xs)
            ###########################
            with torch.cuda.amp.autocast(enabled=False):
                L_seg = criterion(pred, masks[:,0,:,:]) \
                           + dice_loss(F.softmax(pred, dim=1).float(),
                                       F.one_hot(masks[:,0,:,:], model.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
            L_global = L_seg 
            grad_scaler.scale(L_global).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running_loss += L_seg.item()
        
        wandb.log({"loss": running_loss})
        if epoch%10 == 0 :    
            #print('[%d] loss: %.3f' % (epoch + 1, running_loss/len(dataset)))
            torch.save(model.state_dict(), saving_root)
    wandb.finish()


def printer(dataloader_source, device, text) :
    real_batch = next(iter(dataloader_source))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training images "+text+" - a batch")
    plt.imshow(np.transpose(vutils.make_grid(real_batch['image'].to(device)[:8], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
def IoU(res, mask) :
    inter = np.logical_and(res, mask)
    union = np.logical_or(res, mask)
    iou_score = np.sum(inter) / np.sum(union) 
    return iou_score


def postprocessing(res_seg) :   
    res_seg[res_seg < 0.5] = 0
    res_seg[res_seg > 0.5] = 1
    #where_0 = np.where(res_seg == 0)
    #where_1 = np.where(res_seg == 1)  
    #res_seg[where_0] = 1
    #res_seg[where_1] = 0 
    return(res_seg)

def eval(model_root, dataset, device, model_arch="new"):
    if model_arch == "new" :
        model = UNet(n_channels=3, n_classes=2).to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "old" :
        model = UNet_old(n_channels=3, n_classes=2, norm_layer="bn").to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "city" :
        tmp = make_model(n_channels = 3,
                   n_classes = 19,
                   lr = 1e-5,
                   dataloader = dataset,
                   epochs = 1000)
        checkpoint = torch.load(model_root, map_location=device) # ie, model_best.pth.tar
        tmp.network.load_state_dict(checkpoint['network'])
        model = tmp.network

    stack_seg = []
    stack_gt = []
    model.eval()
    torch.manual_seed(42)
    
    if model_arch == "new" or model_arch == "old" :
        with torch.no_grad():
            for batch in (dataset) :
                images = batch['image']
                masks = batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                tmp = model(images)
                seg = postprocessing(F.softmax(tmp, dim=1).detach().cpu().numpy())
                stack_seg.append(seg[0,1])
                stack_gt.append(masks[0,1].detach().cpu().numpy())
        return IoU(stack_seg, stack_gt)
    
    if model_arch == "city" :
        total_dice = 0 
        cpt = 0
        with torch.no_grad():
            for batch in dataset:
                batch = map(lambda x: x.to(device, non_blocking=True), batch)
                images, labels = batch
                labels = prepare_labels(labels, model.n_classes, device)
                outputs = model(images)
                outputs = outputs.softmax(dim=1)
                pred = merge_canal(outputs[0].cpu())
                
                dice = multiclass_dice_coeff(labels, outputs).item()
                
                total_dice+=dice
                cpt +=1
        return(total_dice/cpt)
            

    
def BN_adapt(model_root, dataset, device, saving_root, model_arch="new", run_name=None) :
    if model_arch == "new" :
        model = UNet(n_channels=3, n_classes=2).to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "old" :
        model = UNet_old(n_channels=3, n_classes=2, norm_layer="bn").to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "city" :
        model = make_model(n_channels = 3,
                   n_classes = 19,
                   lr = 1e-5,
                   dataloader = dataset,
                   epochs = 1000)
        checkpoint = torch.load(model_root, map_location=device) # ie, model_best.pth.tar
        model.network.load_state_dict(checkpoint['network'])
    
    if model_arch == "new" or model_arch == "old" :
        model.train()
        with torch.no_grad():
            for batch in dataset :
                images = batch['image'].to(device=device, dtype=torch.float32)
                _ = model(images)
        torch.save(model.state_dict(), saving_root)
        
    if model_arch == "city" :
        run_name = run_name
        model.network.train()
        with torch.no_grad():
            for batch in dataset :
                batch = map(lambda x: x.to(device, non_blocking=True), batch)
                images, _ = batch
                _ = model.network(images)
        save(model, list(), list(), "/tmp/", output_dir="/tmp/", run_name=run_name)


def source_normalized_wasserstein(source, target):
    """ source and target of shape (num_samples, distribution_dim). """
    mu_s, sigma_s = source.mean(dim=0), source.T.cov()
    mu_t, sigma_t = target.mean(dim=0), target.T.cov()
    sigma_s_inv = torch.linalg.pinv(sigma_s)
    L_s, V_s = torch.linalg.eigh(sigma_s)
    sigma_s_inv_square_root = V_s @ torch.diag(1 / L_s.sqrt()) @ torch.linalg.inv(V_s)
    try :
        L_t, V_t = torch.linalg.eigh(sigma_t)
    except _LinAlgError :
        print("hummm LinAlgError")
    sigma_t_square_root = V_t @ torch.diag(L_t.sqrt()) @ torch.linalg.inv(V_t)
    I = torch.eye(len(mu_s))  # noqa: E741
    mu_distance = (mu_t - mu_s) @  sigma_s_inv @ (mu_t - mu_s)
    distance_matrix = I + sigma_t @ sigma_s_inv - 2 * sigma_t_square_root @ sigma_s_inv_square_root
    sigma_distance = torch.trace(distance_matrix)
    return mu_distance #+ sigma_distance


def get_latent_space(dataset, model_root, device, model_arch="new") :
    latent_space = torch.tensor([]).to("cpu")
    if model_arch == "new" :
        model = UNet(n_channels=3, n_classes=2).to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "old" :
        model = UNet_old(n_channels=3, n_classes=2, norm_layer="bn").to(device=device)
        model.load_state_dict(torch.load(model_root, map_location=device))
    if model_arch == "city" :
        tmp = make_model(n_channels = 3,
                   n_classes = 19,
                   lr = 1e-5,
                   dataloader = dataset,
                   epochs = 1000)
        checkpoint = torch.load(model_root, map_location=device) # ie, model_best.pth.tar
        tmp.network.load_state_dict(checkpoint['network'])
        model = tmp.network

    model.eval()
    torch.manual_seed(42)
    
    if model_arch == "old" or model_arch == "new" :
        for batch in dataset :
            images = batch['image']
            images = images.to(device=device, dtype=torch.float32)
            tmp = model.down4(model.down3(model.down2(model.down1(model.inc(images)))))
            latent_space = torch.cat((latent_space,tmp.detach().cpu()))
            del images
            del tmp
        return(latent_space)
    
    if model_arch == "city" :
        for batch in dataset :
            batch = map(lambda x: x.to(device, non_blocking=True), batch)
            images, _ = batch
            images = images.to(device=device, dtype=torch.float32)
            tmp = model.down4(model.down3(model.down2(model.down1(model.inc(images)))))
            latent_space = torch.cat((latent_space,tmp.detach().cpu()))
            del images
            del tmp
        return(latent_space)
    

