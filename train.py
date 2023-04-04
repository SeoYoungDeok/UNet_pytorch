import yaml
import torch
from torch.nn.functional import one_hot
import os
from tqdm import tqdm
from data_loader.data_loader import get_dataloader
import model.model as module_model


def main(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"{device=}")
    epochs = config["epoch"]

    train_loader, test_loader = get_dataloader(
        path=config["data_path"], batch_size=config["batch_size"]
    )
    model = getattr(module_model, config["model"])(21).to(device)

    loss_fn = getattr(torch.nn, config["loss"])()
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    # training loop
    for epoch in range(epochs):
        with tqdm(train_loader) as pbar:
            pbar.set_description(f"Epoch : {epoch}")
            sum_loss = 0
            sum_acc = 0
            sum_len = 0
            for imgs, labels in pbar:
                imgs = imgs.to(device)  # B C W H
                labels = labels.to(device=device, dtype=torch.int64)  # B W H
                # B W H V -> B V W H
                labels = one_hot(labels, 21).permute(0, 3, 1, 2).to(dtype=torch.float32)

                optimizer.zero_grad()

                pred = model(imgs)  # B C W H
                loss = loss_fn(pred, labels)

                pred_idx = torch.argmax(pred, dim=1)
                print(torch.unique(pred_idx[0], return_counts=True))
                label_idx = torch.argmax(labels, dim=1)
                print(torch.unique(label_idx[0], return_counts=True))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                sum_loss += loss.item()
                sum_acc += torch.sum(pred_idx == label_idx).item() / imgs.size(-1) ** 2
                sum_len += imgs.size(0)

                pbar.set_postfix(
                    train_loss=f"{sum_loss / sum_len:.3f}",
                    train_acc=f"{sum_acc / sum_len:.3f}",
                )

        with torch.no_grad():
            model.eval()
            with tqdm(test_loader) as pbar:
                pbar.set_description(f"Epoch : {epoch}")
                sum_loss = 0
                sum_acc = 0
                sum_len = 0
                for imgs, labels in pbar:
                    imgs = imgs.to(device)
                    labels = labels.to(device=device, dtype=torch.int64)
                    labels = (
                        one_hot(labels, 21).permute(0, 3, 1, 2).to(dtype=torch.float32)
                    )

                    pred = model(imgs)
                    loss = loss_fn(pred, labels)

                    pred_idx = torch.argmax(pred, dim=1)
                    label_idx = torch.argmax(labels, dim=1)

                    sum_loss += loss.item()
                    sum_acc += (
                        torch.sum(pred_idx == label_idx).item() / imgs.size(-1) ** 2
                    )
                    sum_len += imgs.size(0)

                    pbar.set_postfix(
                        test_loss=f"{sum_loss / sum_len:.3f}",
                        test_acc=f"{sum_acc / sum_len:.3f}",
                    )
            model.train()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "check_point/model_" + str(epoch) + ".pth")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs("check_point", exist_ok=True)
    main(config)
