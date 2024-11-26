import utils.utils as utils
import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import streamlit as st


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config, placeholder, text_placeholder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img_from_pil(config["content_img"], device) 
    style_img_resized = utils.prepare_img_from_pil(config["style_img"], device)

    if config['init_method'] == 'random':
        if config['noise'] == 'white':
            white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
            init_img = torch.from_numpy(white_noise_img).float().to(device)
            
        else:
            gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
            init_img = torch.from_numpy(gaussian_noise_img).float().to(device)

    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        init_img = style_img_resized


    # we are tuning optimizing_img's pixels. (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['content_feature_map_index'], config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img_resized)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(config['iterations']):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                text = f"""total loss={total_loss.item():12.4f}\ncontent_loss={config["content_weight"] * content_loss.item():12.4f}\nstyle loss={config["style_weight"] * style_loss.item():12.4f}\ntv loss={config["tv_weight"] * tv_loss.item():12.4f}"""
                text_placeholder.text(text)
                current_img = optimizing_img.clone().squeeze(0).cpu().numpy()
                current_img = utils.to_image_format(current_img)  # Normalize and convert to uint8

                placeholder.image(current_img, caption=f"Iteration {cnt}", use_container_width=True)
                st.session_state.style_transfer_progress.append(current_img)

    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=config['iterations'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                text = f"""total loss={total_loss.item():12.4f}\ncontent_loss={config["content_weight"] * content_loss.item():12.4f}\nstyle loss={config["style_weight"] * style_loss.item():12.4f}\ntv loss={config["tv_weight"] * tv_loss.item():12.4f}"""
                text_placeholder.text(text)
                current_img = optimizing_img.clone().squeeze(0).cpu().numpy()
                current_img = utils.to_image_format(current_img)  # Normalize and convert to uint8

                placeholder.image(current_img, caption=f"Iteration {cnt}", use_container_width=True)
                st.session_state.style_transfer_progress.append(current_img)
               
            cnt += 1
            return total_loss

        optimizer.step(closure)
