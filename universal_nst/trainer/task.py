import argparse
import time
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pathlib import Path
from ..config import config
from ..model.dataset import UnsupervisedImageFolder
from ..model.vgg import VGGAutoEncoder
from ..utils import get_device, preprocess, imsave

def train(train_loader, model, optim, config):
    
    # Set training mode
    model.train()
    
    device = get_device()
    
    start_time = time.time()

    for epoch in range(config["EPOCHS"]):
        t_loss = 0
        for batch_id, batch in enumerate(train_loader):
            
            step = len(train_loader) * epoch + batch_id
            
            # Zero Gradient
            optim.zero_grad()  


            # Put batch on GPU
            batch = batch.to(device)

            #Get model output
            decoder_output, encoded_img, encoded_decoder_img = model(batch)

            # Calcuate and Track loss

            # Reconstruction loss gets per-pixel loss and averages for each image in the batch
            reconstruction_loss = mse_loss(decoder_output, batch, reduction="sum") / batch.size(0)

            # Feature loss is the per-element loss of the encoded layer for each image in the batch
            feature_loss = mse_loss(encoded_decoder_img, encoded_img, reduction="sum") / batch.size(0)

            # Total loss is a weighted sum of the two losses above, parameterized by a weight lambda
            total_loss = reconstruction_loss + config["LAMBDA"] * feature_loss
            t_loss += total_loss
            # back prop
            total_loss.backward()
            optim.step()

            if (step == 0) or ((step+1) % config["SAVE_MODEL_EVERY"] == 0) or (step==config["EPOCHS"]*len(train_loader) - 1):
                # Print Losses
                print("========Iteration {}/{}========".format(step+1, config["EPOCHS"]*len(train_loader)))
                print("\tReconstruction Loss:\t{:.2f}".format(reconstruction_loss))
                print("\tFeature Loss:\t{:.2f}".format(feature_loss))
                print("\tTotal Loss:\t{:.2f}".format(total_loss))

                # Save Model
                model_name = model.rep_layer + "_autoencoder"
                checkpoint_path = os.path.join(config["SAVE_MODEL_PATH"], model_name, "checkpoints")
                image_sample_path = os.path.join(config["SAVE_MODEL_PATH"], model_name, "samples")
                Path(checkpoint_path).mkdir(parents=True, exist_ok=True) 
                Path(image_sample_path).mkdir(parents=True, exist_ok=True)

                checkpoint_name = "checkpoint_" + str(step+1) + ".pt"
                if step==config["EPOCHS"]*len(train_loader) - 1:
                    checpoint_name = "final.pth"
                chkpoint = {
                    "optim" : optim.state_dict(),
                    "model" : model.state_dict(),
                    "model_rep" : model.rep_layer,
                    "data_loader": train_loader,
                    "epoch" : epoch, 
                    "global_loss" : t_loss
                    }
                torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "most_recent.pt"))
                print("Saved VGGAutoEncoder checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = decoder_output[0]
                imsave(sample_tensor, os.path.join(image_sample_path, f"sample_{step}.jpg")) 
                print("Saved sample tranformed image at {}".format(image_sample_path))
            
        print(f"Loss for Epoch: {t_loss}, took {time.time()-start_time} ms.\n\n")

    return t_loss.item()

def init_model(rep_layer):
    model = VGGAutoEncoder(rep_layer=rep_layer).to(get_device())
    
    # freeze encoder, update decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    return model

def init_dataset(path_to_dataset, batch_size, shuffle):
    dataset = UnsupervisedImageFolder(path_to_dataset, transform=preprocess())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader, dataset

def resume_from_checkpoint(chkpoint):
    
    state_dicts = torch.load(chkpoint)
    model = VGGAutoEncoder(rep_layer=state_dicts["rep_layer"]).to(get_device())
    model.load_state_dict(state_dicts["model"], strict=False)
    
    optimizer = Adam(model.decoder.parameters()) 
    optimizer.load_state_dict(state_dicts["optim"])
    
    return model, optimizer, state_dicts["epoch"], state_dicts["global_loss"]

if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser(description='Training for vgg autoencoders.')
    parser.add_argument('rep_layer', type=str,
                    help='Which AE to train')
    parser.add_argument("learning_rate", type=float, default=config["LEARNING_RATE"])
    parser.add_argument('--batch', nargs="?", type=int, default=config["BATCH_SIZE"])
    args = parser.parse_args()
    
    if args.learning_rate:
        config["LEARNING_RATE"] = args.learning_rate

    if args.batch:
        config["BATCH_SIZE"] = args.batch
    # Get dataset
    train_loader, train_dataset = init_dataset(config["TRAIN_DATASET_PATH"], config["BATCH_SIZE"], shuffle=True)
    
    
    model = init_model(args.rep_layer)
    
    optimizer = Adam(model.decoder.parameters(), lr=config["LEARNING_RATE"])
    
    print("Training job starting. Training VGGAutoEncoder on layer {}.".format(args.rep_layer))
    print("Configs: ", config)
    print("\n\n")
    
    if os.path.exists(os.path.join(config["SAVE_MODEL_PATH"], args.rep_layer, "checkpoints", "most_recent.pt")):
        print("Checkpoint found, resuming from that file")
        chkpoint_path = os.path.join(config["SAVE_MODEL_PATH"], args.rep_layer, "checkpoints", "most_recent.pt")
        train_loader, model, optimizer, epoch, _ = resume_from_checkpoint(chkpoint_path)
        config["EPOCHS"] -= epoch
    
    train(train_loader, model, optimizer, config)