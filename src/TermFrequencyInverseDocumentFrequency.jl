module TermFrequencyInverseDocumentFrequency

using SparseArrays
using DataStructures

import Base.show

export TermMatrix
export select_vocabulary, inverse_document_frequencies, normalize!, calc_tfidf
export default_stop_words

include("stopwords.jl")

"""
    select_vocabulary(corpus; 
        min_document_frequency=1, transform=identity, pattern=r"\\w\\w+\\b", stopwords=Set{String}())

Select words from the corpus based on the rules defined by the key word arguments.
Stop words are excluded from the vocabulary list.
Use `stopwords=default_stop_words()` to get a list of words from Sci-kit Learn. 
"""
function select_vocabulary(corpus::Vector{<:AbstractString}; 
    min_document_frequency::Int=1, transform=identity, pattern::Regex=r"\w\w+\b", stopwords::Set{String}=Set{String}(),
    )
    vocab_df = DefaultDict{String, Int}(0)
    for document in corpus
        words = Set{String}()
        for m in eachmatch(pattern, transform(document))
            word = m.match
            if (word in stopwords)
                continue
            end
            if !(word in words)
                push!(words, word)
                vocab_df[word] += 1
            end
        end
    end
    vocab_df = filter(x->x[2] ≥ min_document_frequency, vocab_df)
    collect(keys(vocab_df))
end

struct TermMatrix
    matrix::SparseMatrixCSC
    features::Dict
end

function Base.show(io::IO, tm::TermMatrix)
    print(io, "TermMatrix(")
    xnnz = nnz(tm.matrix)
    m, n = size(tm.matrix)
    print(io, "matrix=", m, "×", n, " ", typeof(tm.matrix), " with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
    print(io, ", features=")
    show(io, tm.features)
    print(io, ")")
end

"""
    TermMatrix(corpus, vocab; pattern=r"\\w\\w+\\b")
    TermMatrix(matrix, features)

Calculate the term matrix of nwords × ndocuments for a corpus.
Each column corresponds to a document and each row to the number of occurances of a given term in each document.

`tm.matrix` => a sparse matrix representation of the term matrix.

`tm.features` => a dictionary which maps terms to row numbers.
"""
function TermMatrix(corpus::Vector{<:AbstractString}, vocab::Vector{String}; transform=identity, pattern::Regex=r"\w\w+\b")
    ndocuments = length(corpus)
    nwords = length(vocab)
    sort!(vocab)
    features = Dict{String, Int}()
    for (idx, word) in enumerate(vocab)
        features[word] = idx
    end
    rows = Int[]
    columns = Int[]
    values = Int[]
    for (j, document) in enumerate(corpus)
        frequencies = document_frequencies(features, transform(document); pattern=pattern)
        for (word, value) in frequencies
            push!(rows, features[word])
            push!(columns, j)
            push!(values, value)
        end
    end
    term_counts = sparse(rows, columns, values, nwords, ndocuments)
    TermMatrix(term_counts, features)
end

function document_frequencies(features::Dict, document::AbstractString; pattern::Regex)
    frequencies = DefaultDict{String, Int}(0)
    for m in eachmatch(pattern, document)
        word = m.match
        if haskey(features, word)
            frequencies[word] += 1
        end
    end
    frequencies
end

"""
    inverse_document_frequencies(term_matrix)

Calculate the smoothed inverse document frequencies as `log.((1 + ndocuments)./(1 .+ df)) .+ 1` where `df` is the document frequency.
The constant `1` in the numerator and denominator corresponds to an extra document containing each word in the vocabulary once, and prevents division by zero errors.
"""
inverse_document_frequencies(tm::TermMatrix) = inverse_document_frequencies(tm.matrix)

function inverse_document_frequencies(X::SparseMatrixCSC{Int, Int})
    ndocuments = size(X, 2)
    df = vec(sum(X .> 0, dims=2))
    idf = log.((1f0 + ndocuments)./(1f0 .+ df)) .+ 1f0
    idf
end

"""
    calc_tfidf(X, idf)

Calculate the term frequency - inverse document infrequency matrix. Optimised for sparse matrices. 
Mathematically equivalent to `X .* idf` followed by `normalize!`
"""
calc_tfidf(tm::TermMatrix, idf::Vector{<:AbstractFloat}) = calc_tfidf(tm.matrix, idf)

function calc_tfidf(X::SparseMatrixCSC{Int, Int}, idf::Vector{T})  where T <: AbstractFloat
    tf_idf = similar(X, T)
    tf_idf_vals = nonzeros(tf_idf)
    vals = nonzeros(X)
    rows = rowvals(X) 
    for j = 1:size(X, 2)
        for k in nzrange(X, j)
            i = rows[k]
            tf_idf_vals[k] = vals[k] * idf[i]
        end
    end
    normalize!(tf_idf)
    tf_idf
end

"""
    normalize!(X)

Normalize each column of a sparse matrix such that `sum(X[:, j].^2) == 1`.
"""
function normalize!(X::SparseMatrixCSC{<:T}) where T <: AbstractFloat
    vals = nonzeros(X)
    for j in 1:size(X, 2)
        ks =  nzrange(X, j)
        vals[ks] .*= one(T) /sqrt(sum(vals[ks].^2, dims=1)[1])  
    end
end

function normalize(X::SparseMatrixCSC{<:AbstractFloat})
    Xnorm = copy(X)
    normalize!(Xnorm)
    Xnorm
end

end # module TermFrequencyInverseDocumentFrequency