import torch

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step':0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index':0,
            'train_loss':[],
            'train_acc':[],
            'val_loss':[],
            'val_acc':[],
            'test_loss':-1,
            'test_acc':-1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args:
    :param model:
    :param train_state:
    :return:
    """
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tml, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step']+=1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_best_val'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(predict, rel_vec):
    """

    :param predict: [b_s, 1]
    :param rel_vec: [b_s, 1]
    :return:
    """
    one_hot_y = torch.zeros(rel_vec.shape[0], 10).scatter_(1, rel_vec, 1)
    y = torch.max(one_hot_y, 1)[1]
    correct = torch.eq(predict, y)
    acc = correct.sum().float() / float(correct.data.size()[0])
    acc = acc.to('cpu').data.numpy()*100
    return acc

