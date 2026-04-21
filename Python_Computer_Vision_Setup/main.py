import torch
import setup_data, training_testing_engine, model_creation, utils

from torchvision import transforms


def main():
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir = "data/desert101/train"
    test_dir = "data/desert101/test"

    data_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]
    )

    train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model = model_creation.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    training_testing_engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 )

    utils.save_model(model=model,
                     target_dir="models",
                     model_name="desert_classifier.pth")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()