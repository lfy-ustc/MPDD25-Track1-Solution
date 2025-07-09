from datetime import datetime
import os
import json
import time
import argparse
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from train_val_split import train_val_split1, train_val_split2,get_data
# from models.our.my_model import ourModel
# from models.our.my_model_pro_v1 import ourModel
from models.our.my_model_pro import ourModel
from dataset_my import *
from utils.logger import get_logger
import numpy as np

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def eval(model, val_loader, device):
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    person_id_list = []
    with torch.no_grad():
        for data in val_loader:
            for k, v in data.items():
                if k != 'person_id' and k != 'chunk_id':
                    data[k] = v.to(device)
            model.set_input(data)
            model.test()
            emo_pred = model.emo_pred.argmax(dim=1).cpu().numpy()
            emo_label = data['emo_label'].cpu().numpy()
            total_emo_pred.append(emo_pred)
            total_emo_label.append(emo_label)
            person_id_list.extend(data['person_id'])

    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)

    emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)
    class_counts = np.bincount(total_emo_label)  # Get the sample size for each category
    sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)  # Calculate weights for each sample to avoid division by zero errors
    emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)

    emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')
    emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)

    return total_emo_label,total_emo_pred,emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm


def train_model(train_json, test_json, model, audio_path='', video_path='', max_len=5,
                best_model_name='best_model.pth', seed=None):
    """
    This is the traing function
    """
    logger.info(f'personalized features used：{args.personalized_features_file}')
    num_epochs = args.num_epochs
    device = args.device
    print(f"device: {device}")
    model.to(device)

    # split training and validation set
    # data = json.load(open(train_json, 'r'))
    if opt.train_test:
        # train_data,train_category_count = get_data(train_json,args.track_option)
        # val_data,val_category_count = get_data(test_json,args.track_option)
        test_data,test_category_count = get_data(test_json,args.track_option)

    # else:
    #     if args.track_option=='Track1':
    #         train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=opt.val_ratio, random_seed=seed)
    #     elif args.track_option=='Track2':
    #         train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=opt.val_ratio,
    #                                                                                     seed=seed)
    if args.track_option=='Track1':
        train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=opt.val_ratio, random_seed=seed)
    elif args.track_option=='Track2':
        train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=opt.val_ratio,
                                                                                    seed=seed)
    if opt.longtail:
        class_counts = [0 for i in range(args.labelcount)]
        for key in train_category_count.keys():
            if key >= args.labelcount:
                class_counts[args.labelcount-1] += train_category_count[key]
                class_counts[args.labelcount-1] += val_category_count[key]
            else:
                class_counts[key] += train_category_count[key]
                class_counts[key] += val_category_count[key]
        logger.info("Class counts:\n%s", json.dumps(class_counts, indent=4))
        # class_counts = torch.tensor(class_counts).to(device)
        model.set_cls_num_list(class_counts)
    
    train_dataset = AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path)
    val_dataset = AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path)
    test_dataset = AudioVisualDataset(test_data, args.labelcount, args.personalized_features_file.replace('Training', 'Testing', 1), max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path.replace('Training', 'Testing', 1), video_path=video_path.replace('Training', 'Testing', 1))
    
    train_loader = DataLoader(train_dataset
        , batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset
        , batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset
        , batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info('The number of training samples = %d' % len(train_loader.dataset))
    logger.info('The number of val samples = %d' % len(val_loader.dataset))
    logger.info('The number of test samples = %d' % len(test_loader.dataset))

    best_emo_acc = 0.0
    best_emo_f1 = 0.0
    best_emo_epoch = 1
    best_emo_cm = []

    for epoch in range(num_epochs):
        model.train(True)
        total_loss = 0
        # if epoch == reweight_epoch and opt.longtail:
        #     model.set_criterion()
        total_emo_CE = 0
        total_emo_kl = 0
        total_EmoF_CE = 0
        total_phq = 0
        for i, data in enumerate(train_loader):
            for k, v in data.items():
                if k != 'person_id' and k != 'chunk_id':
                    data[k] = v.to(device)
            model.set_input(data)
            model.optimize_parameters(epoch)

            losses = model.get_current_losses()
            for key in losses.keys():
                total_loss += losses[key]
            
            total_emo_CE += losses['emo_CE']
            total_EmoF_CE += losses['EmoF_CE']
            if 'emo_kl' in losses.keys():
                total_emo_kl += losses['emo_kl']
            if 'phq_loss' in losses.keys():
                total_phq += losses['phq_loss']

        avg_loss = total_loss / len(train_loader)
        # evaluation
        label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader,
                                                                                                device)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.10f}, "
                    f"emo_CE: {total_emo_CE:.3f}, emo_kl: {total_emo_kl:.3f}, EmoF_CE: {total_EmoF_CE:.3f}, phq_loss: {total_phq:.3f}, "
                    f"Weighted F1: {emo_f1_weighted:.10f}, Unweighted F1: {emo_f1_unweighted:.10f}, "
                    f"Weighted Acc: {emo_acc_weighted:.10f}, Unweighted Acc: {emo_acc_unweighted:.10f}")
        logger.info('Confusion Matrix:\n{}'.format(emo_cm))
        label,pred,emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, test_loader,
                                                                                                device)
        logger.info(f"Test Weighted F1: {emo_f1_weighted:.10f}, Test Unweighted F1: {emo_f1_unweighted:.10f}, "
                    f"Test Weighted Acc: {emo_acc_weighted:.10f}, Test Unweighted Acc: {emo_acc_unweighted:.10f}")
        logger.info('Test Confusion Matrix:\n{}'.format(emo_cm))
        if emo_f1_weighted > best_emo_f1:
            cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
            best_emo_f1 = emo_f1_weighted
            best_emo_f1_unweighted = emo_f1_unweighted
            best_emo_acc = emo_acc_weighted
            best_emo_acc_unweighted = emo_acc_unweighted
            best_emo_cm = emo_cm
            best_emo_epoch = epoch + 1
            best_model = model
            # best_model_name = f"best_model_{cur_time}_{best_emo_epoch}.pth"
            save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")
    logger.info(f"Best Weighted F1: {best_emo_f1:.4f}, Best Unweighted F1: {best_emo_f1_unweighted:.4f}, "
                f"Best Weighted Acc: {best_emo_acc:.4f}, Best Unweighted Acc: {best_emo_acc_unweighted:.4f}.")
    logger.info('Confusion Matrix 1:\n{}'.format(best_emo_cm))

    # output results to CSV
    csv_file = f'{opt.log_dir}/{opt.name}.csv'
    formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])
    header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
             f"batch_size,num_epochs,feature_max_len,lr," \
             f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"
    result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{args.splitwindow_time},{args.labelcount},{args.audiofeature_method},{args.videofeature_method}," \
                   f"{args.batch_size},{args.num_epochs},{opt.feature_max_len},{opt.lr:.6f}," \
                   f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"
    file_exists = os.path.exists(csv_file)
    # Open file (append if file exists, create if it doesn't)
    with open(csv_file, mode='a') as file:
        if not file_exists:
            file.write(header + '\n')
        file.write(result_value + '\n')

    return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MDPP Model")
    parser.add_argument('--labelcount', type=int, default=3,
                        help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True,
                        help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True,
                        help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True,
                        help="Root path to the program dataset")
    parser.add_argument('--train_json', type=str, required=False,
                        help="File name of the training JSON file")
    parser.add_argument('--personalized_features_file', type=str,
                        help="File name of the personalized features file")
    parser.add_argument('--audiofeature_method', type=str, default='mfccs',
                        choices=['mfccs', 'opensmile', 'wav2vec'],
                        help="Method for extracting audio features.")
    parser.add_argument('--videofeature_method', type=str, default='densenet',
                        choices=['openface', 'resnet', 'densenet'],
                        help="Method for extracting video features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s',
                        help="Time window for splitted features. e.g. '1s' or '5s'")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of epochs to train the model")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to train the model on, e.g. 'cuda' or 'cpu'")
    parser.add_argument('--model', type=str, default='my_model_pro',
                        help="Model to train, e.g. 'my_model_pro' or 'my_model'")
    parser.add_argument('--config_path', type=str, default='config/config.json',
                        help="Path to the config file")
    
    args = parser.parse_args()

    args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')
    args.test_json = os.path.join(args.data_rootpath,'Testing', 'labels', 'Testing_files_label.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')

    config = load_config(args.config_path)
    opt = Opt(config)

    # Modify individual dynamic parameters in opt according to task category
    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr
    # reweight_epoch = opt.reweight_epoch

    # Splice out feature folder paths according to incoming audio and video feature types
    audio_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(args.data_rootpath, 'Training', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    # Obtain input_dim_a, input_dim_v
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]            
            break
    

    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}+{args.track_option}+{args.model}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')
    model = ourModel(opt)

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"
    logger.info("Opt parameters: %s", opt.__dict__)
    logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, "
                f"videofeature_method={args.videofeature_method}")
    logger.info(f"batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")

    # set random seed
    seed = np.random.randint(0, 10000) 
    # seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f"Using random seed: {seed}")

    # training
    train_model(
        train_json=args.train_json,
        test_json=args.test_json,
        model=model,
        max_len=opt.feature_max_len,
        best_model_name=best_model_name,
        audio_path=audio_path,
        video_path=video_path,
        seed=seed
    )
