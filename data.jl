const uddir = "/scratch/users/okirnap/ud-treebanks-v2.2"
const erenay = "/scratch/users/okirnap/ku-dependency-parser2/erenay"

const engtrn = "$uddir/UD_English-EWT/en_ewt-ud-train.conllu"
const engdev = "$uddir/UD_English-EWT/en_ewt-ud-dev.conllu"
const englm = "$lmdir/english_chmodel.jld"

const trtrn = "$uddir/UD_Turkish-IMST/tr_imst-ud-train.conllu"
const trdev = "$uddir/UD_Turkish-IMST/tr_imst-ud-dev.conllu"
const trerenay = "$erenay/tr_imst-dev.erenay"
const trlm  = "$lmdir/turkish_chmodel.jld"

const arabtrn =  "$uddir/UD_Arabic-PADT/ar_padt-ud-train.conllu"
const arabdev =  "$uddir/UD_Arabic-PADT/ar_padt-ud-dev.conllu"
const araberenay = "$erenay/ar_padt-dev.erenay"
const arablm = "$lmdir/arabic_chmodel.jld"

const butrn =  "$uddir/UD_Bulgarian-BTB/bg_btb-ud-train.conllu"
const budev =  "$uddir/UD_Bulgarian-BTB/bg_btb-ud-dev.conllu"
const bulm = "$lmdir/bulgarian_chmodel.jld"

const hutrn =  "$uddir/UD_Hungarian-Szeged/hu_szeged-ud-train.conllu"
const hudev =  "$uddir/UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu"
const huerenay = "$erenay/hu_szeged_dev.erenay"
const hulm  = "$lmdir/hungarian_chmodel.jld"

const ettrn = "$uddir/UD_Estonian-EDT/et_edt-ud-train.conllu"
const etdev = "$uddir/UD_Estonian-EDT/et_edt-ud-dev.conllu"
const etlm  = "$lmdir/estonian_chmodel.jld"

const hrtrn = "$uddir/UD_Croatian-SET/hr_set-ud-train.conllu"
const hrdev = "$uddir/UD_Croatian-SET/hr_set-ud-dev.conllu"
const hrlm  = "$lmdir/croatian_chmodel.jld"

const eltrn = "$uddir/UD_Greek-GDT/el_gdt-ud-train.conllu"
const eldev = "$uddir/UD_Greek-GDT/el_gdt-ud-dev.conllu"
const elerenay = "$erenay/el_gdt-dev.erenay"
const ellm  = "$lmdir/greek_chmodel.jld"

const enltrn = "$uddir/UD_English-LinES/en_lines-ud-train.conllu"
const enldev = "$uddir/UD_English-LinES/en_lines-ud-dev.conllu"
const enllm = englm



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
