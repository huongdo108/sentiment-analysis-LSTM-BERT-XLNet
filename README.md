# Sentiment Analysis with Bidirectional LSTM, BERT, and XLNet

## Project Overview

In this project, I implemented sentiment analysis on Movie Review Dataset by using three different deep learning architectures: Bidirectional LSTM, transfer learning with BERT and XLNet.

Data can be download from <a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie Review Dataset</a>. 

## How to run

<a href="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/read_data.ipynb">Process data function</a> is used to read in, process, and save processed data in <a href="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/tree/master/processed_data">Processed data folder</a>

<a href="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/LSTM.ipynb">LSTM notebook</a>

<a href="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/BERT.ipynb">BERT notebook</a>

<a href="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/XLNet.ipynb">XLNet notebook</a>

## LSTM 

### LSTM overview

Long Short-Term memory (LSTM) is an artificial recurrent neural network (RNN) architecture. 

### Limitation of previous models (vanilla RNN) and Naive Bayes 

Naive Bayes makes strong assumption of the independence of words in a sentence so it does not capture the semantic relationships between words in a sentence.

In RNN, information travels through time in which the information produced at the previous time point is used as input information for the next time point. Vanilla RNN fails to address the problem of long-term dependencies as it is unable to learn to connect the information when the gap between the relevant information and the place where it is needed is big. This problem is called Vanishing Gradient. 


### Key components of LSTM

LSTM can address the above mentioned Vanishing Gradient problem. LSTM has the chain like structure of RNN but different repeating module structure. It has 4 interacting neural network layers instead of 1 layer in RNN. [24]

A LSTM unit includes a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals while the gates regulate the information flow in and out of the cell. 

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/rnn.PNG" align="centre">
Figure 1. RNN cell [24]

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/LSTM.PNG" align="centre">
Figure 2. LSTM cell [24]

## BERT 

### BERT overview

BERT, a Bidirectional Encoder Representations from Transformers, a language representation model proposed by a Google AI team. "BERT is designed to pretrain deep bidirectional representations from unlabeled text
by jointly conditioning on both left and right context in all layers. As a
result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of
tasks, such as question answering and language inference, without substantial task-specific architecture modifications" [2].

According to its published paper, BERT obtains new improved results
on eleven natural processing tasks, including improving GLUE score by
7.7 % to 80.5 %, increasing MultiNLI acurracy score by 4.6 % to 86.7 %,
improving SQuAD v1.1 question answering Test F1 by 1.5 % to 93.2 %,
and improving SQuAD v2.0 Test F1 by 5.1 % to 83.1 % [2].

### Limitations of previous models (Word2Vec, Glove, ElMo and OpenAI) 

Word2Vec and Glove do not solve the problem of Polysemy, same words
but having different meanings for different contexts. [14]. Thus, those
same words will have similar embedding vector representation regardless of their meanings under Word2Vec and Glove. Besides, the neural
network architecture of Word2Vec is very shallow, a single hidden layer
fully connected, [17], which means that the information that Word2Vec
can capture during training is limited.

ElMo, a contextualized word embeddings, is able to solve the problem of
Polysemy by utilizing bidirectional Long Short Term Memory (LSTM), an
improved recurrent neural network that can learn long-term dependencies among words in input sentences [13]. Elmo utilizes a concatenation
of two independent LSTMs: forward left-to-right and backward right-toleft in pre-training [2]. This architecture, however, does not take into
account the previous words and subsequent words simultaneously.

OpenAI overcomes the shortcomings of LSTM by building on the stateof-the-art Transformer architecture. Thus, OpenAI while has a simpler
architecture so that it can train faster compared to other LSTM-based
models, it is able to learn more complex patterns by utilizing the Atention mechanism of Transformer [14]. However, OpenAI just utilizes only
the Decoder of Transformer to predict the next word given the previous
context (a left-to-right language model).

### Key components of BERT

It is observed from figure 4 that BERT is truly a deep bidirectional language model by taking into account the context of previous words and
subsequent words simultaneously from the first to the last layer of the
neural network, compared to the unidirectional language model OpenAI
and the shallow bidirectional language model achieved by concatenation
of two independent passes ELMo.

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/bert_compare.PNG" align="centre">

Figure 3. ’Differences in pre-training model architectures. BERT uses a bidirectional
Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the
concatenation of independently trained left-to-right and right-toleft LSTMs to
generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers.’
[2]

BERT is able to achieve this bidirectional language model by the following improvements. (In the following part, we just focus on the improvements that are in concerned with our project’s objective, English word
prediction in a sentence)

**Multi-layer bidirectional Transformer Encoder** Transformer, a model architecture proposed by a Google AI team in the paper ’Attention is all you -->
need’ [18]. Transformer avoids the Recurrence-Based models (RNN and
LSTM) by introducing Attention Mechanism which together works with
positional embeddings to eliminate the sequential input, but still be able
to capture the positional embedding, in order to enable parallel computation [18].

**Masked Language Model** randomly masks some input sentence’s tokens
and uses the context information of the previous and subsequent words
simultaneously to predict the original words of those masked tokens [2].
Bert uses Masked Language Model as one of its two pre-training tasks
(the other task is Next Sentence Prediction which is not under the concern
of this project). During Masked Language Model pre-training task, 15%
of the total tokens in the input sentence are chosen to be the target of
prediction. In order to reduce the discrepancy between pre-training and
fine tuning, those 15% chosen tokens are not always being masked, but
are either: 

* replaced by a [MASK] token (80% of the time)

* replaced by a random token (10% of the time)

* kept intact (10% of the time)

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/Bert_architecture.PNG" align="centre">
Figure 4. ’An illustrated BERT architecture in predicting masked word’ [1]

## XLNet

### XLNet overview

XLNet is the-state-of-the-art language model proposed by the Google AI
Brain Team at Carnegie Mellon University. "XLNet, a generalized autoregressive pretrained method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all the permutations of
the factorization order and (2) overcomes the limitations of BERT thanks
to its autoregressive formulation" [22]. According to its published paper,
XLNet outperforms BERT on 20 tasks (including 7 GLUE language understanding tasks [20]), including natural language inference, sentiment
analysis, question answering, and document ranking.

### Limitations of previous models (auto-regressive model and BERT)

A classic autoregressive language model predicts a target word by only
depending on that target word’s previous words or subsequent words as
the model is trained to encode only unidirectional context, either forward
or backward. This type of model is proved to be not effective at modelling
deep bidirectional contexts which are required by downstream tasks [22].

BERT overcomes the unidirectional context shortcoming of autoregressive model as we already mentioned in BERT’s section that BERT is
trained in the way that, given the input token sequence, a portion of to-
kens are chosen to be masked, and the model is trained to recover the
masked tokens given the bidirectional context, the surrounding words of
masked tokens. Despite BERT’s significant improvements from previous
language models, it still has some limitations:

* The [MASK] tokens which BERT uses during pretraining phase are not
present in real data at fine tuning phase for downstream tasks, leading
to the discrepancy of pretrain-fine tune [22].

* BERT uses Transformer’s Auto Encoding objective instead of Auto Regressive objective, so it is not able to model the joint probability using
the product rule as in autoregressive model, as a result, it might not be
powerful enough for some generative tasks [22].

* BERT assumes the predicted tokens are independent of each other as
BERT predicts them in parallel. As a result, the model does not learn
to handle dependencies among predicting masked tokens. This denies
the fact that there exists long-range, high-order dependency in natural
language [22].

### Key components of XLNet

This section focuses on the components that helps XLNet to outperform
other models in English text prediction.

**Permutations of the factorization order**  By introducing the concept called
permutations of the factorization order, XLNet is able to leverage the best
of both autoregressive model and BERT model while avoid their limitations. Permutation operations make the context of each position consists of tokens from both left and right without the need of masking [22].
In other words, the model is forced to model bidirectional dependencies
among all combinations of inputs using permutation, which contrasts to
traditional language models that learn dependencies in one direction, and
which is different from BERT that uses masking [9]. Thus, XLNet can
maximize the expected log likelihood of a sequence with respect to all
possible permutations. As a result, XLNet can avoid the discrepancy of
pretrain-finetune that BERT suffers. Furthermore, the autoregressive
objective that XLNet uses also provides a way to use the product rule
for factorizing the joint probability of the predicted tokens, thus XLNet
is able to predict all words in random permuted order while BERT just
predicts only the masked (15% ) tokens simultaneously. Therefore, XLNet can avoid the assumption of the independence of predicted tokens in
BERT [22]. The permutation does not permute the input sequence order,
but the factorization order instead [22].

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/XLNet_permutation.PNG" align="centre">
Figure 5. Illustration of the permutation language modelling objective for predicting x3 given the same input sequence x but with different factorization orders [22]

The example below shows how XLNet and BERT capture the independence among predicted tokens [New] and [York] given the context [is,a,city].

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/XLNet_dependency.PNG" align="centre">
Figure 6. XLNet captures more dependency pairs than BERT [3]

**Two-stream attention mask** XLNet relies on the new state-of-the-art twostream attention mask and position encoding to achieve the permutation.
Traditional attention mask and position encoding are two of the core components of the Transformers architecture which is introduced by a Google
team in the paper “Attention is all you need” [18]. Positional encoding
captures the positional information of input sequences. Attention mask
gives different attention to each input token given its context [18]. For
language models using Transformers architecture, the entire embedding
including the positional embedding of the predicted token is masked out.
Thus, the model is cut off from the knowledge with respect to that token’s
position This can pose problem especially if that token is positioned at the
starting of sentence which has different distribution from other sentence’s
positions [9]). XLNet’s two-stream attention mask can address this problem by introducing two kinds of attentions: query stream attention and
content stream attention.

* **The query stream attention**is the contextual representation for each
token which contains information from contextual words and the predicted token’s current position (not information from the predicted token’s content). The model is trained to predict each token in the sentence by using information from this query stream attention [9].

* **The content stream attention** is the content representation containing both the positional embedding and token embedding. The content
stream attention is used as input to query stream attention and not the
other way around [9].

<img src="https://github.com/huongdo108/sentiment-analysis-LSTM-BERT-XLNet/blob/master/images/2streams-attention.PNG" align="centre">
Figure 7. (a): Content stream attention, which is the same as the standard self-attention.
(b): Query stream attention, which does not have access information about the
content x. (c): Overview of the permutation language modelling training with
two-stream attention [22].

**Use Transformer-XL as base model** XLNet incorporates 2 key ideas from
Transformer-XL into pretraining which are the segment-level recurrence
mechanism and relative-position encoding in order to improve the performance especially in modelling long text sequence. Transformer-XL is introduced by a Google Team in the paper ‘Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context’ [23].

* **The segment-level recurrence mechanism** seeks to solve the problem that occurs in traditional Transformer. Vanilla Transformer takes
fixed-length sequences as inputs so that computations can be paralleled.
In case of very long text, this can be infeasible because of memory constraint. Vanilla Transformer does not have hidden states like Recurrent
Neural Network which can chunk the long text into fixed-length segments and feed to the model one chunk at a time without resetting the
hidden states between chunks. Transformer XL achieves this by adding
recurrence but at the segment level. Hidden state sequence computed
for the previous segment is cached and reused as an extended context
when the model processes the next new segment. Thus, the segmentlevel recurrence mechanism can effectively process very long text while
preventing text fragmentation issues [22].

* **Relative-positional encoding** seeks to solves the problem occurred
by segment-level recurrent mechanism. Given a sequence, what happens to the positional embedding of the first word in the previous segment? Transformer-XL uses an embedding to encode the relative distance between words instead of using embedding that represents the
absolute position of a word. [22].

As BERT uses Transformer’s encoder as its base model, by incorporating
these two above methods into its architecture, XLNet also outperforms
BERT in modelling long text sequence.

## References

[1] Jay Alammar. The illustrated bert, elmo, and co. (how nlp cracked transfer
learning). URL: http://jalammar.github.io/illustrated-bert/, 2018.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT:
pre-training of deep bidirectional transformers for language understanding.
CoRR, abs/1810.04805, 2018.

[3] ESSEN.AI. What is xlnet and how does it work? URL: https://blog.ess
en.ai/what-is-xlnet-and-how-does-it-work/, 2019.

[4] Peter A Heeman. Pos tags and decision trees for language modeling. In
1999 Joint SIGDAT Conference on Empirical Methods in Natural Language
Processing and Very Large Corpora, 1999.

[5] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

[6] Huggingface. Pretrained models. URL: https://huggingf ace.co/trans
formers/pretrained_models.html, 2020.

[7] Sheri Hunnicutt and Johan Carlberger. Improving word prediction using
markov models and heuristic methods. Augmentative and Alternative Communication, 17(4):255–264, 2001.

[8] Jonathan Frederic Jason Grout and Sylvain Corlay. ipywidgets: Interactive widgets for the jupyter notebook. URL: https://github.com/jupyterwidgets/ipywidgets, 2020.

[9] Keita Kurita. Paper dissected: “xlnet: Generalized autoregressive pretraining for language understanding” explained. URL: https://mlexplained.
com/2019/06/30/paper-d issected -xlnet-generalized -autoregressiv
e-pretraining-for-language-understanding-explained/, 2019.

[10] Gregory W Lesher, Bryan J Moulton, D Jeffery Higginbotham, et al. Effects
of ngram order and training text size on word prediction. In Proceedings of
the RESNA’99 Annual Conference, pages 52–54. Citeseer, 1999.

[11] Tomáš Mikolov, Anoop Deoras, Stefan Kombrink, Lukáš Burget, and Jan
Cernock ˇ y. Empirical evaluation and combination of advanced language `
modeling techniques. In Twelfth Annual Conference of the International
Speech Communication Association, 2011.

[12] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Cernock ˇ y, and Sanjeev `
Khudanpur. Recurrent neural network based language model. In Eleventh
annual conference of the international speech communication association,
2010.

[13] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word
representations. CoRR, abs/1802.05365, 2018.

[14] MOHD SANAD ZAKI RIZVI. Demystifying bert: A comprehensive guide to
the groundbreaking nlp framework. URL: https://www.analyticsvid hy
a.com/blog/2019/09/d emystifying-bert-groundbreaking-nlp-framew
ork/, 2019.

[15] Xin Rong. word2vec parameter learning explained. CoRR, abs/1411.2738,
2014.

[16] Aman Rusia. Padding text to help transformer-xl and xlnet with short
prompts as proposed by aman rusia. URL: https://gith ub.com/h ugging
face/transformers/blob/master/examples/run/generation.py, 2020.

[17] Greg Corrado Jeffrey Dean Tomas Mikolov, Kai Chen. Efficient estimation
of word representations in vector space. CoRR, abs/1301.3781, 2013.

[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you
need. CoRR, abs/1706.03762, 2017.

[19] Tonio Wandmacher and Jean-Yves Antoine. Methods to integrate a language model with semantic information for a word prediction component.
CoRR, abs/0801.4716, 2008.

[20] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and
Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform
for natural language understanding. URL: https:// openreview.net/p
df ?id=rJ4km2R5t7, 2019.

[21] Deutscher Wortschatz. English corpora. URL: https://wortschatz.unileipzig.de/en/download/english, 2018.

[22] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for
language understanding. In Advances in neural information processing
systems, pages 5754–5764, 2019.

[23] Yiming Yang Jaime Carbonell Quoc V. Le2 Ruslan Salakhutdinov Zihang Dai,
Zhilin Yang. Transformer-xl: Attentive language models beyond a fixedlength context. CoRR, abs/1901.02860, 2018.

[24] Understanding LSTM networks. URL: https://colah.github.io/posts/2015-08-Understanding-LSTMs/


