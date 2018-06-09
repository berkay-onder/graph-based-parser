const engtrn = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu"
const engdev = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu"
const englm = "$lmdir/english_chmodel.jld"

const trtrn = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-train.conllu"
const trdev = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-dev.conllu"
const trlm  = "$lmdir/turkish_chmodel.jld"






# TODO: refactor Omer's code
function load_data(lm, trnfile, devfile)
    d, v, c = depmain("--lmfile $lm --datafiles $trnfile $devfile")
    return d, v, c
end


function bucketize(sents)::Dict
    buckets = Dict()
    for sent in sents
        push!(get!(buckets, length(sent), []), sent)
    end
    return buckets
end


function minibatch(buckets::Dict, mbsize::Integer; 
                   remaining=false, shuffle=true, 
                   minlength=2, maxlength=128, use_tokens=true)
    if shuffle
        for k in keys(buckets)
            shuffle!(buckets[k])
        end
    end
    mbatches = []
    bucket_keys = Iterators.filter(k->minlength<=k<=maxlength,
                                   keys(buckets))
    for k in bucket_keys
        bucket = buckets[k]
        mbs = use_tokens ? div(mbsize, k) : mbsize
        for i = 1:mbs:length(bucket)
            ending = i+mbs-1 
            if ending>length(bucket) 
                !remaining && break
                ending = length(bucket)
            end
            push!(mbatches, bucket[i:ending])
        end
    end
    return mbatches
end
