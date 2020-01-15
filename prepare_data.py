from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import random
import bert_utils
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_jsonl", None,
    "Gzipped files containing NQ examples in Json format, one per line.")

flags.DEFINE_string("output_tfrecord", None,
                    "Output tf record file with all features extracted.")
flags.DEFINE_string("vocab_file",None,
                    "Output tf record file with all features extracted.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

import codecs
def get_examples(input_jsonl_pattern):
  for input_path in tf.gfile.Glob(input_jsonl_pattern):
    with codecs.open(input_file, encoding='utf-8', mode='r') as input_file:
      for line in input_file:
        yield bert_utils.create_example_from_jsonl(line)


def main(_):
  examples_processed = 0
  num_examples_with_correct_context = 0
  creator_fn = bert_utils.CreateTFExampleFn(is_training=FLAGS.is_training)

  instances = []
  for example in get_examples(FLAGS.input_jsonl):
    for instance in creator_fn.process(example):
      instances.append(instance)
    if example["has_correct_context"]:
      num_examples_with_correct_context += 1
    if examples_processed % 100 == 0:
      tf.logging.info("Examples processed: %d", examples_processed)
    examples_processed += 1
    if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
      break
  tf.logging.info("Examples with correct context retained: %d of %d",
                  num_examples_with_correct_context, examples_processed)

  random.shuffle(instances)
  with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for instance in instances:
      writer.write(instance)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()