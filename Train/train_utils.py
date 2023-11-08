import torch
import numpy as np
from tqdm import tqdm
from DSAL import DSAL
import os
import albumentations as A

def get_label(label):
    if label == 'unfertilized':
        return 0
    elif label == '_PKCa':
        return 1
    elif label == 'N_KCa':
        return 2
    elif label == 'NP_Ca':
        return 3
    elif label == 'NPK_':
        return 4
    elif label == 'NPKCa':
        return 5
    else:
        return 6
def lambda_transform(x: np.array, **kwargs) -> np.array:
    return x / 255


def transform_image_label(image, label, transform, mean=None, std=None):
    out_image = image.copy()
    out_label = get_label(label)

    if transform is not None:
        augmented = transform(image=out_image)
        out_image = augmented['image']

        if mean is not None and std is not None:
            temp = A.Compose(
                transforms=[
                    A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
                ],
                p=1.0
            )

            augmented = temp(image=out_image)
            out_image = augmented['image']

        # converting the image and mask into tensors

    out_image = torch.from_numpy(out_image).permute(2, 0, 1)
    out_image = out_image.float()
    out_label = torch.tensor(out_label).long()

    return out_image, out_label

def evaluate(model, val_batches, epoch, device, criterion):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0

    for batch in val_batches:
        image, label = batch
        # label = label.type(torch.int64)
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            # outputs = outputs.type(torch.float32)
            loss = criterion(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

    loss = total_loss / total
    accuracy = total_correct / total
    return loss, accuracy, f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.8f}, Accuracy: {accuracy:6.8f}\n'


def train(model, 
        criterion,
        optimizer,
        val_batches,
        train_images,
        train_transform,
        mean,
        std,
        labels,
        out_text_name,
        model_save_dir,
        last_save_name,
        num_workers=20,
        batch_size=32,
        epochs=50,
        epoch_step=10,
        gamma=0.75,
        input_message=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'):

    train_dsal = DSAL(images=train_images,
                    read_and_transform_function=transform_image_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    num_processes=num_workers,
                    yml=labels,
                    transform=train_transform,
                    mean=mean,
                    std=std)

    train_dsal.start()

    counter = 0
    batches_per_epoch = train_dsal.num_batches // epochs
    epoch = 0
    total = 0
    total_correct = 0
    total_loss = 0

    best_loss = 1000
    best_accuracy = 0
    best_epoch = 0

    f = open(os.path.join(os.path.dirname(model_save_dir), out_text_name), 'w')

    f.write(input_message)


 # scheduler: optimizer, step size, gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epoch_step, gamma)
    for i in tqdm(range(train_dsal.num_batches)):

        if counter == batches_per_epoch:
            total_loss = total_loss / total
            accuracy = total_correct / total
            message = f'Training --- Epoch: {epoch}, Loss: {total_loss:6.8f}, Accuracy: {accuracy:6.8f}\n'
            current_loss, current_accuracy, print_output = evaluate(model=model,
                                                                    val_batches=val_batches,
                                                                    epoch=epoch,
                                                                    criterion=criterion,
                                                                    device=device)
            print(message)
            print(print_output)
            f.write(message)
            f.write(print_output)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch

                torch.save(model, model_save_dir)

            if current_loss < best_loss:
                best_loss = current_loss

            message = f'Best epoch: {best_epoch}, Best Loss: {best_loss:6.8f}, Best Accuracy: {best_accuracy:6.8f}\n'
            print(message)
            f.write(message)
            model.train()

            total = 0
            total_correct = 0
            total_loss = 0
            epoch += 1
            counter = 0
            scheduler.step()

        image, label = train_dsal.get_item()
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()
        total += image.size(0)
        _, predictions = outputs.max(1)
        total_correct += (predictions == label).sum()
        total_loss += loss.item() * image.size(0)

        counter += 1

    total_loss = total_loss / total
    accuracy = total_correct / total

    message = f'Training --- Epoch: {epoch}, Loss: {total_loss:6.8f}, Accuracy: {accuracy:6.8f}\n'
    loss, accuracy, eval_message = evaluate(model=model,
                                        val_batches=val_batches,
                                        epoch=epoch,
                                        criterion=criterion,
                                        device=device)

    print(message)
    print(eval_message)

    f.write(message)
    f.write(eval_message)

    f.close()

    train_dsal.join()

    torch.save(model, last_save_name)


def find_mean_std(test_dsal):
    sum_ = torch.zeros(3)
    sq_sum = torch.zeros(3)
    num_images = 0

    print(f'\n---finding mean and std ---')

    for _ in tqdm(range(test_dsal.num_batches)):
        image, _ = test_dsal.get_item()
        batch = image.size(0)
        sum_ += torch.mean(image, dim=[0, 2, 3]) * batch
        sq_sum += torch.mean(image ** 2, dim=[0, 2, 3]) * batch
        num_images += batch

    mean = sum_ / num_images
    std = ((sq_sum / num_images) - mean ** 2) ** 0.5

    return mean, std

    