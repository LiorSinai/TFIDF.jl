# TermFrequencyInverseDocumentFrequency.jl 
## Term Frequency - Inverse Document Frequency (TF-IDF)

[![Build Status](https://github.com/LiorSinai/PackageTemplate.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LiorSinai/PackageTemplate.jl/actions/workflows/CI.yml?query=branch%3Amain)

A model based on term frequency and inverse document frequency.
- Term frequency: number of times a term/word/token appears in a document.
- Document frequency: 1 if appears in a document; 0 otherwise. 

These are calculated per document and the results are stored in a `TermMatrix` where each column `j` is for a document, each row `i` is for a term, and `term_matrix[i, j]` is the number of times term `i` appears in document `j`. This is most commonly a sparse matrix so for efficiency it is represented in the `SparseMatrixCSC` format from `SparseArrays`.

The document frequency is then `df=sum(term_matrix .> 0, dims=2)`. The inverse document frequency is the reciprocal of this. Here it is also normalized and shifted by one: `idf=log.((1 + ndocuments)./(1 .+ df)) .+ 1`. 

These two values multiplied together give the TF-IDF: `term_matrix .* idf`. For efficiency, this calculation is not done using a broadcast but rather by iterating over the sparse matrix arrays. A model can then use the TF-IDF as the input feature. 

## Usage

```Julia
using TermFrequencyInverseDocumentFrequency
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is a third one.",
    "Is this the first document?",
]
words = select_vocabulary(corpus; transform=lowercase)
term_matrix = TermMatrix(corpus, words; transform=lowercase)
idf = inverse_document_frequencies(term_matrix)
tfidf = calc_tfidf(term_matrix, idf)
# 9×4 SparseMatrixCSC{Float32, Int64} with 20 stored entries:
#  ⋅         ⋅        0.531146   ⋅ 
# 0.453491  0.674531   ⋅        0.453491
# 0.560151   ⋅         ⋅        0.560151
# 0.370758  0.275737  0.277174  0.370758
#  ⋅         ⋅        0.531146   ⋅
#  ⋅        0.528392   ⋅         ⋅
# 0.453491  0.337266   ⋅        0.453491
#  ⋅         ⋅        0.531146   ⋅
# 0.370758  0.275737  0.277174  0.370758
```

Logistic regression:
```Julia
using Flux
using Flux: onecold, onehotbatch, logitcrossentropy
using StatsBase: mean

loss(model, x::AbstractMatrix, y::AbstractMatrix) = logitcrossentropy(model(x), y)
opt = ADAM()
model = Dense(length(words), nlabels, identity)
opt_state = Flux.setup(opt, model)
data =  Flux.DataLoader((tfidf, one_hot_labels); batchsize=32, shuffle=true)
Flux.train!(loss, model, data, opt_state)
acc = mean(onecold(model(tfidf)) .== onecold(one_hot_labels))
```

Most important words:
```Julia
for label in 1:nlabels
    top_words = [words[idx] for idx in sortperm(model.weight[label, :], rev=true)]
    println(label, ": ", top_words[1:10])
end
```

## Installation

Download the GitHub repository (it is not registered). Then in the Julia REPL:
```Julia
julia> ] # enter package mode
(@v1.x) pkg> dev path\\to\\TermFrequencyInverseDocumentFrequency
julia> using Revise # for dynamic editing of code
julia> using TermFrequencyInverseDocumentFrequency
```

## Related Packages

- [MLJText](https://github.com/JuliaAI/MLJText.jl)
- [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl)