using TFIDF
using Test
using SparseArrays

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is a third one.",
    "Is this the first document?",
]

@testset "select vocabulary" begin
    words = select_vocabulary(corpus; transform=lowercase)
    expected = [
        "and",
        "document",
        "is",
        "first",
        "one",
        "the",
        "third",
        "this",
        "second",
    ]
    @test issetequal(words, expected)
end

@testset "calc_tfidf " begin
    X = sparse([1, 1, 2, 2, 3, 3], [1, 3, 1, 4, 2, 3], [3, 2, 2, 5, 1, 2], 3, 4)
    v = [2.0, 3.0, 3.0]
    Y_expected = [
        0.707107   0     0.5547   0
        0.707107   0     0        1.0
        0          1.0   0.83205  0
    ]
    Y = calc_tfidf(X, v)
    @test isapprox(Y, Y_expected; atol=1e-5)

    nrow, ncol = 5, 7
    X = sprand(Int, nrow, ncol, 0.3)
    v = rand(nrow)
    Y1 = X .* v
    normalize!(Y1)
    Y2 = calc_tfidf(X, v)
    @test Y1 == Y2
end

@testset "TFIDF " begin
    words = [
        "and",
        "document",
        "is",
        "first",
        "one",
        "second",
        "the",
        "third",
        "this",
    ]
    term_matrix = TermMatrix(corpus, words; transform=lowercase)
    features = Dict(
        "and" => 1,
        "document" => 2,
        "first" => 3,
        "is" => 4,
        "one" => 5,
        "second" => 6,
        "the" => 7,
        "third" => 8,
        "this" => 9,
    )
    @test term_matrix.features == features
    matrix = [
        0 0 1 0;
        1 2 0 1;
        1 0 0 1;
        1 1 1 1;
        0 0 1 0;
        0 1 0 0;
        1 1 0 1;
        0 0 1 0;
        1 1 1 1;
    ]
    @test term_matrix.matrix == matrix

    expected_idf = Float32[
        1.9162908
        1.2231436
        1.5108256
        1.0
        1.9162908
        1.9162908
        1.2231436
        1.9162908
        1.0
    ]
    idf = inverse_document_frequencies(term_matrix)
    @test idf ≈ expected_idf

    expected_tfidf = [
        0         0         0.531146  0
        0.453491  0.674531  0         0.453491
        0.560151  0         0         0.560151
        0.370758  0.275737  0.277174  0.370758
        0         0         0.531146  0
        0         0.528392  0         0
        0.453491  0.337266  0         0.453491
        0         0         0.531146   0
        0.370758  0.275737  0.277174  0.370758
    ]
    tfidf = calc_tfidf(term_matrix, idf)
    @test isapprox(tfidf, expected_tfidf; atol=1e-5)
    
    tfidf_broadcast = term_matrix.matrix .* idf
    normalize!(tfidf_broadcast)
    @test isapprox(tfidf, tfidf_broadcast; atol=1e-8)
end
