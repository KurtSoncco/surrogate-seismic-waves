#!/bin/bash
# Contact-feature arm. Not started from this file until P128n128 and P128s1
# have exited and the seed gap below has been filled in from their best
# checkpoints. Neck 32, from scratch, seed 0. One change against P128.
#
# Decision rule, fixed before any contact-arm number is read.
# Comparators are best_model.pt of P128 and of P128s1, the checkpoints chosen
# by the selection score (mean of IID and manifold 0.1-10 Hz log-L1).
# For a readout R, Delta(R) = |R(P128) - R(P128s1)| on those checkpoints.
# The arm clears R only when it improves on the better of the two by more than
# 2*Delta(R). This bar applies to the selection score, dipping, and three-layer.
# The same checkpoints are also read at matched epochs. The saved epoch is the
# luckiest draw of a score that moves about ±0.001 from epoch to epoch, so a
# best-checkpoint gap inside that jitter is not a separate win.
#
# Delta is unknown until P128s1 stops. Record it here before using the arm:
#   selection  ________
#   dipping    ________
#   three      ________
#   bar = 2 * Delta

echo "contact arm is not launched from this script yet" >&2
exit 1
