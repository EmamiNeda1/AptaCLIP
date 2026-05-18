import argparse
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={
            "interaction": "binary_crossentropy",
            "triplet_output": cosine_triplet_loss(0.3)
        },
        loss_weights={
            "interaction": 1.0,
            "triplet_output": 1.0
        },
        metrics={
            "interaction": [
                "accuracy",
                keras.metrics.AUC(name="auc")
            ]
        }
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True
            )
        ]
    )

    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_test, y_test, "Test")

    model.save(args.output)

    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        default="data/train_set.csv"
    )

    parser.add_argument(
        "--test",
        type=str,
        default="data/test_set.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/aptaclip_model.keras"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    main(args)
