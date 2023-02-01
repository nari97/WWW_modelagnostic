# A Model-Agnostic Method to Interpret Link Prediction Tasks Performed by Knowledge Graph Embeddings

## Abstract 
Link prediction assigns plausibility scores to unseen triples in a
knowledge graph using an input partial triple. Models compute
plausibility scores based on embeddings, numerical vectors that
encode the semantics of the graph. Performance metrics like mean
rank are useful to compare models side by side, but do not shed
light on their behavior. Interpreting the link prediction task is thus a
must to advance the field. Current methods have mainly focused on
single predictions or other tasks different than link prediction. Since
knowledge graph embedding methods are diverse, interpretation
methods that are applicable only to certain machine learning approaches cannot be used. 
In this paper, we propose a model-agnostic method for interpreting the link prediction task as a whole. Triples
deemed plausible by a model are materialized in a new knowledge
graph. We mine Horn rules from such graph to succinctly represent
it. Mined rules are accompanied by precision and recall measurements, which we combine using F𝛽 score to quantify interpretation
accuracy. To increase interpretation accuracy, we study two approximations to the hard problem of merging rules. Our experiments
show that our macro interpretation accuracy serves to compare
diverse models side by side, and that descriptive statistics of our
interpretation accuracy help detect strengths and weaknesses of
models. Our qualitative study shows that several models globally
capture the expected semantics of certain predicates.

## Datasets
Three sets of data were used for this paper. 
1. The embedding models that were trained by [Kelpie](https://github.com/AndRossi/Kelpie) (ComplEx, ConvE, TuckER) can be found as [.pt files](https://figshare.com/s/ede27f3440fe742de60b). These are
to be extracted into the kelpie_trained_models folder in the Kelpie directory.
2. The embedding models that were trained by [TuckER](https://github.com/ibalazevic/TuckER) can be found as [.pt files](). These are
to be extracted into the tucker_trained_models folder in the Tucker directory.
3. The results folder from our experiments can be found as a [.zip file](). These are to be extracted in the Interpretibility folder.
