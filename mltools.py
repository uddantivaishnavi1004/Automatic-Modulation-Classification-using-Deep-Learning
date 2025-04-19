import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import csv

# Show loss curves
def show_history(history):
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig('figure/total_loss.png')
    plt.close()

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history['accuracy'], label='train_acc')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_acc')
    plt.legend()    
    plt.savefig('figure/total_acc.png')
    plt.close()

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    epoch=history.epoch
    np_train_acc=np.array(train_acc)
    np_val_acc=np.array(val_acc)
    np_train_loss=np.array(train_loss)
    np_val_loss=np.array(val_loss)
    np_epoch=np.array(epoch)
    np.savetxt('train_acc.txt',np_train_acc)
    np.savetxt('train_loss.txt',np_train_loss)
    np.savetxt('val_acc.txt',np_val_acc)
    np.savetxt('val_loss.txt',np_val_loss)

# Calculate confusion matrix
def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    for k in range(0, Y.shape[0]):
        i = list(Y[k, :]).index(1)
        j = int(np.argmax(Y_hat[k, :]))
        conf[i, j] = conf[i, j] + 1

    for i in range(0, n_classes):
        row_sum = np.sum(conf[i, :])
        if row_sum == 0:
            confnorm[i, :] = 0  # Avoid division by zero
        else:
            confnorm[i, :] = conf[i, :] / row_sum

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong

# Calculate accuracy for each SNR
def calculate_acc_cm_each_snr(Y, Y_hat, Z, classes=None, save_figure=True, min_snr=-20):
    Z_array = Z if Z.ndim == 1 else Z[:, 0]  # Ensure Z is 1D
    snrs = sorted(list(set(Z_array)))
    acc = np.zeros(len(snrs))
    acc_mod_snr = np.zeros((len(classes), len(snrs)))

    for i, snr in enumerate(snrs):
        if snr < min_snr:
            continue  # Skip SNRs below the threshold
        Y_snr = Y[np.where(Z_array == snr)]
        Y_hat_snr = Y_hat[np.where(Z_array == snr)]
        cm, right, wrong = calculate_confusion_matrix(Y_snr, Y_hat_snr, classes)
        acc[i] = round(1.0 * right / (right + wrong), 3)
        acc_mod_snr[:, i] = calculate_acc_at1snr_from_cm(cm)

    # Plot overall accuracy vs. SNR
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc, label='test_acc')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on All Test Data")
    plt.legend()
    plt.grid()
    plt.savefig('figure/acc_overall.png')
    plt.show()

    # Save accuracy values
    with open('acc111.csv', 'a', newline='') as f0:
        csv_writer = csv.writer(f0)
        for acc_value in acc:
            csv_writer.writerow([acc_value])

    # Save accuracy data
    with open('acc_overall_128k_on_512k_wts.dat', 'wb') as fd:
        pickle.dump(('128k', '512k', acc), fd)

    # Plot accuracy for each modulation type
    dis_num = 6
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        beg_index = g * dis_num
        end_index = min((g + 1) * dis_num, acc_mod_snr.shape[0])
        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Modulation Type")
        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
        plt.legend()
        plt.grid()
        if save_figure:
            plt.savefig(f'figure/acc_with_mod_{g + 1}.png')
        plt.show()

    # Save modulation accuracy data
    with open('acc_for_mod_on_1m_wts.dat', 'wb') as fd:
        pickle.dump(('128k', '1m', acc_mod_snr), fd)