# November 24, 2016

Training loss on a tiny data set (sanity check) decays to near zero very quickly,
but interestingly, it spikes back up periodically to over 2.0. Further, when we
try to run inference on a single image *from the training set*, we get results
that are completely wrong. This is unexpected.

Couple things that could be going wrong.

1. We did not regenerate the data set after adding GO and PAD to the character
   set. Note: we should also add END at this time, before regenerating.
2. The order of the data used during training is *not* randomized, which could
   lead to some pathological training problems due to the particular sequence
   we are using to train.


## Take 1: Regenerate data set.

Making sure the PAD / GO tokens were correct, and regenerating the tiny dataset,
we see proper convergence properties on the training data. Test data is still
completely nonsensical, of course. There are still small spikes once in a while
in the training loss, but not nearly as drastic.


# November 25, 2016

A few things should be fixed, likely in this order:

1. Add the STOP token to the end of every example. DONE.
2. Randomize the sampling order, using bucketing over output sequence length
   and possibly also input image size.
3. Training should go over the full length of the output sequence.

# November 29, 2016

* We really need a 'dynamic_attention_decoder' using logic like in the 'dynamic_rnn'.
  This might exist out there somewhere.
* For spatial padding, we will probably want to move the meaningful section around
  in the padded image. Not sure that this is possible with tf.batch() and
  dynamic_pad=True.

# December 3, 2016

Fully dynamic version of this model in TF is working. It uses the 'raw_rnn' function.

Other thoughts:

* Consider using "teacher forcing" (using the 'true' output as input to the next iteration) with curriculum learning (slowly decreasing the frequency of teacher forcing).
* Add appropriate metrics again so that TensorBoard is informative.
* Set up infra for hyperparameter optimization.
* Good monitoring of hyperparameter trials. Can we do this in TensorBoard?
* Visualization of decoding process.
* Consolidate preprocessing during training vs. testing
* Inference server should accept an actual image.
* Shuffle training data (and create full-sized data set?)
