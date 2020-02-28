from argparse import Namespace
import os
import torch
from general_util import set_seed_everywhere, handle_dirs, make_embedding_matrix
from data_util.data_utils import generate_batches
from data_util.dataset import SenDataset
from preprocess import preprocess_data
from Layers.config import Config
from Layers.acnn import ACNN
import torch.optim as optim
from tqdm import tqdm
from loss import DistanceLoss
from train_utils import make_train_state, update_train_state, compute_accuracy

args = Namespace(
    # Data and path hyper parameter
    sen_csv='data/dataset/sen_with_pos_splits.csv',
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/myacnn/acnn_model",
    # Model hyper parameter
    glove_filepath='data/glove/glove.6B.100d.txt',
    use_glove=True,
    word_embed_size=100,
    pos_embed_size=25,
    win_size=3,  # k
    kernel_size=3,  # 卷积核大小
    hidden_size=1000,  # d_c
    relation_dim=1000,
    loss_margin=1,
    # Training hyper parameter
    seed=1337,
    learning_rate=0.01,
    dropout_p=0.5,
    batch_size=128,
    num_epochs=100,
    early_stopping_criteria=5,
    weight_decay=1e-5,
    # Runtime option
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)

# Initializations ######################################
sen_df = preprocess_data()
if args.reload_from_files:
    # training from a checkpoint
    dataset = SenDataset.load_dataset_and_load_vectorizer(sen_df,
                                                          args.vectorizer_file,
                                                          args.win_size)
else:
    dataset = SenDataset.load_dataset_and_make_vectorizer(sen_df,
                                                          args.win_size)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

# Use GloVe or randomly initialized embedding
if args.use_glove:
    words = vectorizer.sent_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                       words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None
all_y = [v for v in vectorizer.rel_vocab._token_to_idx.values()]
config = Config.setconfig(args, len(vectorizer.sent_vocab), len(vectorizer.rel_vocab), dataset._max_seq_length, all_y, embeddings)

acnn = ACNN(config)
acnn = acnn.to(args.device)
optimizer = optim.SGD(acnn.parameters(), lr=args.learning_rate, weight_decay=0.0001)  # optimize all rnn parameters
loss_func = DistanceLoss(margin=args.loss_margin,
                         rel_emb_size=args.relation_dim,
                         rel_vocab_size=len(vectorizer.rel_vocab),
                         all_y=all_y
                         )
train_state = make_train_state(args)
epoch_bar = tqdm(desc='training routine',
                 total=args.num_epochs,
                 position=0)

dataset.set_split('train')
train_bar = tqdm(desc='split=train',
                 total=dataset.get_num_batches(args.batch_size),
                 position=1,
                 leave=True)
dataset.set_split('val')
val_bar = tqdm(desc='split=val',
               total=dataset.get_num_batches(args.batch_size),
               position=1,
               leave=True)
try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index
        dataset.set_split('train')
        batch_generator = generate_batches(dataset=dataset,
                                           batch_size=args.batch_size,
                                           device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        acnn.train()
        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:
            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()
            # step 2. compute the output
            wo, predict,rel_emb_weight = acnn(x_in=batch_dict['out_vec'],
                               pos1=batch_dict['d1'], pos2=batch_dict['d2'],
                               e1_vec=batch_dict['e1_vec'], e2_vec=batch_dict['e2_vec'],
                               rel_vec=batch_dict['rel_vec'])
            # step 3. compute the loss
            loss = loss_func(wo, rel_emb_weight, batch_dict['rel_vec'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # step 4. use loss to produce gradients
            loss.backward()
            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            if batch_index%100==0:
                print(predict.to('cpu').data.numpy())
            acc_t = compute_accuracy(predict, batch_dict['rel_vec'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc,
                                  epoch=epoch_index)
            train_bar.update()
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)
        # Iterate over val dataset
        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('val')
        batch_generator = generate_batches(dataset,
                                           batch_size=args.batch_size,
                                           device=args.device)
        running_loss = 0.
        running_acc = 0.
        acnn.eval()
        for batch_index, batch_dict in enumerate(batch_generator):
            # step 2. compute the output
            wo, predict,rel_emb_weight = acnn(x_in=batch_dict['out_vec'],
                               pos1=batch_dict['d1'], pos2=batch_dict['d2'],
                               e1_vec=batch_dict['e1_vec'], e2_vec=batch_dict['e2_vec'],
                               rel_vec=batch_dict['rel_vec'])
            # step 3. compute the loss
            loss = loss_func(wo, rel_emb_weight, batch_dict['rel_vec'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute the accuracy
            acc_t = compute_accuracy(predict, batch_dict['rel_vec'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            val_bar.set_postfix(loss=running_loss, acc=running_acc,
                                epoch=epoch_index)
            val_bar.update()
            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            train_state = update_train_state(args=args, model=acnn,
                                             train_state=train_state)
            if train_state['stop_early']:
                break
            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
except KeyboardInterrupt:
    print("Exiting loop")

# compute the loss & accuracy on the test set using the best available model

acnn.load_state_dict(torch.load(train_state['model_filename']))

acnn = acnn.to(args.device)

loss_func = DistanceLoss(margin=args.loss_margin,
                         rel_emb_size=args.rel_emb_size,
                         rel_vocab_size=len(vectorizer.rel_vocab),
                         all_y=all_y
                         )
dataset.set_split('test')
batch_generator = generate_batches(dataset,
                                   batch_size=args.batch_size,
                                   device=args.device)
running_loss = 0.
running_acc = 0.
acnn.eval()
for batch_index, batch_dict in enumerate(batch_generator):
    # step 2. compute the output
    wo, predict, rel_emb_weight = acnn(x_in=batch_dict['out_vec'],
                                       pos1=batch_dict['d1'], pos2=batch_dict['d2'],
                                       e1_vec=batch_dict['e1_vec'], e2_vec=batch_dict['e2_vec'],
                                       rel_vec=batch_dict['rel_vec'])
    # step 3. compute the loss
    loss = loss_func(wo, rel_emb_weight, batch_dict['rel_vec'])
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)
    # compute the accuracy
    acc_t = compute_accuracy(predict, batch_dict['rel_vec'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss: {};".format(train_state['test_loss']))
print("Test Accuracy: {}".format(train_state['test_acc']))
