# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss


@register_criterion('label_smoothed_cross_entropy_with_classifier_loss')
class LabelSmoothedCrossEntropyCriterionWithClassifierLoss(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, classifier_lambda):
        super().__init__(task, sentence_avg, label_smoothing)
        self.classifier_lambda = classifier_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--classifier-lambda', default=1.0, type=float, metavar='D',
                            help='weight for the classifier loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        classifier_loss = None
        # Compute classifier loss only for training set and non dummy batches.
        if 'is_shuffle' in sample and sample['is_shuffle'] is not None:
            classifier_loss, classifier_nll_loss = self.compute_classifier_loss(model, net_output, sample)

        if classifier_loss is not None:
            logging_output['classifier_loss'] = utils.item(classifier_loss.data)
            logging_output['classifier_nll_loss'] = utils.item(classifier_nll_loss.data)
            loss += self.classifier_lambda * classifier_loss

        return loss, sample_size, logging_output

    def compute_classifier_loss(self, model, net_output, sample, reduce=True):
        classifier_logits = [net_output[1]['classifier_logits']]
        lprobs = model.get_normalized_probs(classifier_logits, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_classifier_targets(sample, net_output).view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        classifier_loss_sum = utils.item(sum(log.get('classifier_loss', 0) for log in logging_outputs))
        classifier_nll_loss_sum = utils.item(sum(log.get('classifier_nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('classifier_loss', classifier_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('classifier_nll_loss', classifier_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
