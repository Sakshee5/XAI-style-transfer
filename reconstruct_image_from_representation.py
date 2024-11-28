import utils.utils as utils
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
import numpy as np
import streamlit as st
import time

def make_tuning_step(model, optimizer, target_representation, content_feature_maps_index, style_feature_maps_indices, content):

    def tuning_step(optimizing_img):
        set_of_feature_maps = model(optimizing_img)
        if content:
            current_representation = set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        else:
            current_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices]

        loss = 0.0
        if content:
            loss = torch.nn.MSELoss(reduction='mean')(target_representation, current_representation)
        else:
            for gram_gt, gram_hat in zip(target_representation, current_representation):
                loss += (1 / len(target_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), current_representation

    return tuning_step

def reconstruct_image_from_representation(config, representation_placeholder, video_placeholder, text_placeholder_1, text_placeholder_2):
    
    uploaded_image = config['content_img'] if config['content_img'] else config['style_img']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = utils.prepare_img_from_pil(uploaded_image, device) 
    
    if config['noise'] == 'white':
        white_noise_img = np.random.uniform(-90., 90., img.shape).astype(np.float32)
        init_img = torch.from_numpy(white_noise_img).float().to(device)
        
    else:
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)

    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['content_feature_map_index'], config['model'], device)

    set_of_feature_maps = neural_net(img)

    if config['content_img']:
        target_content_representation = set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)

        # Display feature maps in the first column
        display_feature_maps(target_content_representation, representation_placeholder, text_placeholder_1)
    else:
        target_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
        display_gram_matrices(target_style_representation, style_feature_maps_indices_names, representation_placeholder, text_placeholder_1)

    target_representation = target_content_representation if config['content_img'] else target_style_representation

    content = True if config['content_img'] else False

    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,))
        tuning_step = make_tuning_step(neural_net, optimizer, target_representation, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], content)

        for it in range(config['iterations']):
            loss, _ = tuning_step(optimizing_img)
            with torch.no_grad():
                text_placeholder_2.write(f'loss={loss:10.8f}')
                current_img = optimizing_img.clone().squeeze(0).cpu().numpy()
                current_img = utils.to_image_format(current_img)  # Normalize and convert to uint8

                video_placeholder.image(current_img, caption=f"Iteration {it}", use_container_width=True)

                if config['content_img']:
                    st.session_state.content_reconstruct.append(current_img)
                else:
                    st.session_state.style_reconstruct.append(current_img)
             

    elif config['optimizer'] == 'lbfgs':
        cnt = 0

        def closure():
            nonlocal cnt
            optimizer.zero_grad()
            
            # Calculate the loss
            if content:
                current_representation = neural_net(optimizing_img)[content_feature_maps_index_name[0]].squeeze(axis=0)
                loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_representation)
            else:
                current_set_of_feature_maps = neural_net(optimizing_img)
                current_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(current_set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
                loss = sum((1 / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
                            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation))

            # Backward pass
            loss.backward()

            # Log the loss and gradients
            with torch.no_grad():
                text_placeholder_2.write(f'loss={loss.item()}')
             
                current_img = optimizing_img.clone().squeeze(0).cpu().numpy()
                current_img = utils.to_image_format(current_img)  # Normalize and convert to uint8

                video_placeholder.image(current_img, caption=f"Iteration {cnt}", use_container_width=True)
                if config['content_img']:
                    st.session_state.content_reconstruct.append(current_img)
                else:
                    st.session_state.style_reconstruct.append(current_img)
                cnt += 1

            return loss

        optimizer = LBFGS((optimizing_img,), max_iter=config['iterations'], line_search_fn='strong_wolfe')

        optimizer.step(closure)

def display_feature_maps(feature_maps, placeholder, text_placeholder_1):
    num_of_feature_maps = feature_maps.size()[0]
    text_placeholder_1.write(f'Number of feature maps: {num_of_feature_maps}')
    time.sleep(1)

    for i in range(len(feature_maps)):
        feature_map = feature_maps[i].to('cpu').numpy()
        feature_map = np.uint8(utils.get_uint8_range(feature_map))

        if i<5:
            placeholder.image(feature_map, caption=f"Feature Map {i}", use_container_width=True)
            time.sleep(1)

        st.session_state.feature_maps.append(feature_map)
        

def display_gram_matrices(gram_matrices, style_feature_maps_indices_names, placeholder, text_placeholder):
    num_of_gram_matrices = len(gram_matrices)
    text_placeholder.write(f'Number of Gram matrices: {num_of_gram_matrices}')
    time.sleep(1)

    for i in range(num_of_gram_matrices):
        gram_matrix = gram_matrices[i].squeeze(axis=0).to('cpu').numpy()
        gram_matrix = np.uint8(utils.get_uint8_range(gram_matrix))
        
        placeholder.image(gram_matrix, caption=f'Gram matrix from layer {style_feature_maps_indices_names[1][i]}', use_container_width=True)
        time.sleep(1)
        st.session_state.gram_matrices.append(gram_matrix)

