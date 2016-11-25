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
completely nonsensical, of course.
