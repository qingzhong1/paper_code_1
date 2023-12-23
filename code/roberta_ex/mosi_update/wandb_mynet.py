from __future__ import absolute_import, division, print_function
import wandb
wandb.init(project="mosi_mynet_50",name='wandb_50')
wandb.watch_called = False
from transformers import AdamW, get_cosine_schedule_with_warmup
import argparse
import transformers
import os
from sklearn.metrics import confusion_matrix
import numpy
import torch
from sklearn.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, cached_path
#from model import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from my_model import BertForSequenceClassification_three, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from my_utils import *
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import numpy as np


path_train='D:\\cmudata\\CMU-MultimodalSDK-master\\examples\\mmdatasdk_examples\\full_examples\\cmumosi\\train.pkl'
path_dev='D:\\cmudata\\CMU-MultimodalSDK-master\\examples\\mmdatasdk_examples\\full_examples\\cmumosi\\dev.pkl'
path_test='D:\\cmudata\\CMU-MultimodalSDK-master\\examples\\mmdatasdk_examples\\full_examples\\cmumosi\\test.pkl'
# 单GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def fetch_model_optim_sched(num_train_optimization_steps, device):
    model = BertForSequenceClassification_three(hidden_dropout_prob=0.1, hidden_size=768, num_labels=1, fusion_dim=5)
    model=model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-05)
    learning_rate: 2e-05
    num_epochs: 5
    gradient_accumulation_step: 1
    warmup_proportion=0.1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )

    return model, optimizer, scheduler
"""BERT finetuning runner."""
def main(i):
    parser = argparse.ArgumentParser(description='MOSI Sentiment Analysis')

    ## Required parameters
    parser.add_argument("--data_dir", default='D:\\cmudata\\CMU-MultimodalSDK-master\\examples\\mmdatasdk_examples\\full_examples\\cmumosi\\', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='D:\\A_L_V\\Cross-Modal-BERT-master\\pre-trained BERT', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='Multi', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='D:\\A_L_V\\Cross-Modal-BERT-master\\Cross-Modal-BERT-master\\CM-BERT_output_three', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.'store_true'")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=24, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=24, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=24, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.5e-5")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=11111,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    wandb.config.update(args)
    processors = {
        "multi": PgProcessor,
    }

    num_labels_task = {
        "multi": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 2
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    seed_num = np.random.randint(1, 10000)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_num)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    '''train_audio, valid_audio, test_audio = pickle.load(
        open('D:\\A_L_V\\Cross-Modal-BERT-master\\data\\audio\\MOSI_cmu_audio_CLS.pickle', 'rb'))'''
    train_audio,train_audio_length=A_V_data(path_train,args.max_seq_length)
    train_audio=torch.tensor(train_audio)
    valid_audio,v_length=A_V_data(path_dev,args.max_seq_length)
    valid_audio=torch.tensor(valid_audio)
    test_audio,test_length=A_V_data(path_test,args.max_seq_length)
    test_audio=torch.tensor(test_audio)
    train_vision,train_vision_length=A_V_data(path_train,args.max_seq_length,mode='vision')
    valid_vision, valid_vision_length = A_V_data(path_dev, args.max_seq_length,mode='vision')
    test_vision, test_vision_length = A_V_data(path_test, args.max_seq_length, mode='vision')
    model, optimizer, scheduler=fetch_model_optim_sched(num_train_optimization_steps,device)
    wandb.watch(model, log="all")
    max_acc = 0
    min_loss = 100
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_train_audio = torch.tensor(train_audio, dtype=torch.float32)
        all_train_vision = torch.tensor(train_vision, dtype=torch.float32)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_train_audio,all_train_vision, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Evaluate for each epcoh
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        all_valid_audio = torch.tensor(valid_audio, dtype=torch.float32, requires_grad=True)
        all_valid_vision=torch.tensor(valid_vision,dtype=torch.float32, requires_grad=True)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float32)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_valid_audio,all_valid_vision, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        all_test_audio = torch.tensor(test_audio, dtype=torch.float32, requires_grad=True)
        all_test_vision = torch.tensor(test_vision, dtype=torch.float32, requires_grad=True)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float32)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_test_audio,
                                  all_test_vision)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
        result_epoch=[]
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            train_accuracy = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            train_pred_list = []
            train_truth_list=[]
            dev_pred_list=[]
            dev_truth_list=[]
            predict_list = []
            truth_list = []
            text_attention_list = []
            fusion_attention_list = []
            with tqdm(train_dataloader, desc="Iteration") as tepoch:
                for step, batch in enumerate(tepoch):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, all_train_audio,all_train_vision, label_ids = batch
                    loss, logits = model(input_ids, all_train_audio,all_train_vision, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tmp_train_accuracy = accuracy(logits, label_ids)
                    train_accuracy += tmp_train_accuracy
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    train_accuracy_rate = train_accuracy / nb_tr_examples
                    nb_tr_steps += 1
                    for i in range(len(logits)):
                        train_pred_list.append(logits[i])
                        train_truth_list.append(label_ids[i])
                    tepoch.set_postfix(
                        {'Batch': '{:3d}'.format(step), 'Train Loss': ' {:5.4f}'.format(loss),
                         'acc': '{:5.4f}'.format(train_accuracy_rate), 'acc_num': '{:5.4f}'.format(tmp_train_accuracy)})
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for input_ids, input_mask, segment_ids, all_valid_audio, all_valid_vision,label_ids in tqdm(eval_dataloader,
                                                                                       desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                all_valid_audio = all_valid_audio.to(device)
                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids, all_valid_audio, all_valid_vision,segment_ids, input_mask, label_ids)
                    # logits,_,_ = model(input_ids,all_valid_audio, segment_ids, input_mask)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
                for i in range(len(logits)):
                    dev_pred_list.append(logits[i])
                    dev_truth_list.append(label_ids[i])

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            dev_predict_list = numpy.array(dev_pred_list).reshape(-1)
            dev_truth_list = numpy.array(dev_truth_list)
            exclude_zero = False
            dev_non_zeros = numpy.array([i for i, e in enumerate(dev_predict_list) if e != 0 or (not exclude_zero)])
            dev_predict_list1 = (dev_predict_list[dev_non_zeros] > 0)
            dev_truth_list1 = (dev_truth_list[dev_non_zeros] > 0)
            # test_loss = test_loss / nb_test_steps
            dev_preds_a7 = numpy.clip(dev_predict_list, a_min=-3., a_max=3.)
            dev_truth_a7 = numpy.clip(dev_truth_list, a_min=-3., a_max=3.)
            dev_acc7 = accuracy_7(dev_preds_a7, dev_truth_a7)
            dev_f_score = f1_score(dev_predict_list1, dev_truth_list1, average='weighted')
            dev_acc = accuracy_score(dev_truth_list1, dev_predict_list1)
            dev_corr = numpy.corrcoef(dev_predict_list, dev_truth_list)[0][1]
            dev_mae = numpy.mean(numpy.absolute(dev_predict_list - dev_truth_list))
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': loss}
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            # Save a trained model and the associated configuration
            if eval_loss < min_loss:
                min_loss = eval_loss
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)


            logger.info("")
            logger.info("***** Running test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.test_batch_size)
            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
            with torch.no_grad():
                for input_ids, input_mask, segment_ids, label_ids, all_test_audio, all_test_vision in tqdm(
                        test_dataloader,
                        desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    all_test_audio = all_test_audio.to(device)
                    all_test_vision = all_test_vision.to(device)
                    with torch.no_grad():
                        tmp_test_loss, logits = model(input_ids, all_test_audio, all_test_vision, segment_ids,
                                                      input_mask, label_ids)
                        logits, text_attention, fusion_attention = model(input_ids, all_test_audio, all_test_vision,
                                                                         segment_ids, input_mask)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    text_attention = text_attention.cpu().numpy()
                    fusion_attention = fusion_attention.cpu().numpy()
                    test_loss += tmp_test_loss.mean().item()

                    for i in range(len(logits)):
                        predict_list.append(logits[i])
                        truth_list.append(label_ids[i])
                        text_attention_list.append(text_attention[i])
                        fusion_attention_list.append(fusion_attention[i])
                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            exclude_zero = False
            non_zeros = numpy.array([i for i, e in enumerate(predict_list) if e != 0 or (not exclude_zero)])
            predict_list = numpy.array(predict_list).reshape(-1)
            truth_list = numpy.array(truth_list)
            predict_list1 = (predict_list[non_zeros] > 0)
            truth_list1 = (truth_list[non_zeros] > 0)
            test_loss = test_loss / nb_test_steps
            test_preds_a7 = numpy.clip(predict_list, a_min=-3., a_max=3.)
            test_truth_a7 = numpy.clip(truth_list, a_min=-3., a_max=3.)
            acc7 = accuracy_7(test_preds_a7, test_truth_a7)
            f_score = f1_score(predict_list1, truth_list1, average='weighted')
            class_report = classification_report(truth_list1, predict_list1)
            print(class_report)
            acc = accuracy_score(truth_list1, predict_list1)
            corr = numpy.corrcoef(predict_list, truth_list)[0][1]
            mae = numpy.mean(numpy.absolute(predict_list - truth_list))
            loss = tr_loss / nb_tr_steps if args.do_train else None
            results = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                       'dev_loss': eval_loss,
                       'dev_acc': dev_acc,
                       'dev_F1': dev_f_score,
                       'dev_mae': dev_mae,
                       'dev_corr': dev_corr,
                       'dev_acc7': dev_acc7,
                'test_loss': test_loss,
                       'global_step': global_step,
                       'loss': loss,
                       'acc': acc,
                       'F1': f_score,
                       'mae': mae,
                       'corr': corr,
                       'acc7': acc7}
            wandb.log({'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                       'dev_loss': eval_loss,
                       'dev_acc': dev_acc,
                       'dev_F1': dev_f_score,
                       'dev_mae': dev_mae,
                       'dev_corr': dev_corr,
                       'dev_acc7': dev_acc7,
                'test_loss': test_loss,
                       'global_step': global_step,
                       'loss': loss,
                       'acc': acc,
                       'F1': f_score,
                       'mae': mae,
                       'corr': corr,
                       'acc7': acc7})
            result_epoch.append(results)
    return result

if __name__ == '__main__':
    for i in range(10):
        os.system(
            'D:\\A_L_V\\multimodal-deep-learning-main\\Cross-Modal-BERT-master\\Cross-Modal-BERT-master\\CM-BERT_output')
        result=main(i)
        print(result)