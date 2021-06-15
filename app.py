import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import data_helpers
from flask import Flask, jsonify, request

app = Flask(__name__)

# Parameters
batch_size = 64


def classify(checkpoint_dir, x_text):

    # Get the vocabulary of the model
    vocab_path = os.path.join("model", "vocab")
    #vocab_path = "/movida/runs/1542030705/vocab"

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
        vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_text)))

    # Create the graph
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(
                "{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name(
                "dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name(
                "output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(
                list(x_test), batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(
                    predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate(
                    [all_predictions, batch_predictions])

            return batch_predictions


@app.route('/', methods=["POST"])
def resultado_frase():

    req_data = request.get_json(force=True)
    text = req_data['text']
    x_text = [text]

    checkpoint_dir = "./model/checkpoints"
    result = classify(checkpoint_dir, x_text)

    predictions_human_readable = np.column_stack((np.array(x_text), result))

    # return jsonify(predictions_human_readable)
    if(result[0] == 1):
        print(x_text)

    return "{}".format(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
