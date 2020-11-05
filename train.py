import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import zipfile

from retinanet.label_encoder import LabelEncoder
from retinanet.prediction_decoder import DecodePredictions
from retinanet.fpn import get_backbone
from retinanet.loss import RetinaNetLoss
from retinanet.model import RetinaNet
from utils.preprocessing import preprocess_data, resize_and_pad_image
from utils.visualization import visualize_detections


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def train():
    model_dir = "retinanet/"
    label_encoder = LabelEncoder()

    num_classes = 80
    batch_size = 2

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    #  set `data_dir=None` to load the complete dataset

    (train_dataset, val_dataset), dataset_info = tfds.load(
        "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
    )

    autotune = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    """
    ## Training the model
    """

    # Uncomment the following lines, when training on full dataset
    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    # val_steps_per_epoch = \
    #     dataset_info.splits["validation"].num_examples // batch_size

    # train_steps = 4 * 100000
    # epochs = train_steps // train_steps_per_epoch

    epochs = 1

    # Running 100 training and 50 validation steps,
    # remove `.take` when training on the full dataset

    model.fit(
        train_dataset.take(100),
        validation_data=val_dataset.take(50),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Change this to `model_dir` when not using the downloaded weights
    weights_dir = "data"

    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
    int2str = dataset_info.features["objects"]["label"].int2str

    for sample in val_dataset.take(2):
        image = tf.cast(sample["image"], dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [
            int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        ]
        visualize_detections(
            image,
            detections.nmsed_boxes[0][:num_detections] / ratio,
            class_names,
            detections.nmsed_scores[0][:num_detections],
        )


if __name__ == "__main__":
    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    keras.utils.get_file(filename, url)

    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("./")

    train()
