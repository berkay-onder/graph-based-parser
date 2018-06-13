include("index.jl")

function read_conllfile(conllfile)
    sentences=[]
    #postaglists=[]
    parentlists=[]
    #deprellists=[]
    conll_input=open(conllfile)
    while !eof(conll_input)
        input_line=readline(conll_input)
        #if startswith(input_line,"# text =")
        while startswith(input_line, "#")
            input_line=readline(conll_input)
        end
            new_sentence=[]
            #new_postaglist=[]
            new_parentlist=[]
            #new_deprellist=[]
            while length(input_line)>0
                conllfields=split(input_line,"\t")
                push!(new_sentence,String(conllfields[2]))
                #push!(new_postaglist,String(conllfields[4]))
                push!(new_parentlist,parse(conllfields[7]))
                #push!(new_deprellist,parse(conllfields[8]))
                input_line=readline(conll_input)
            end
            push!(sentences,new_sentence)
            #push!(postaglists,new_postaglist)
            push!(parentlists,new_parentlist)
            #push!(deprellists,new_deprellist)
        #end
    end
    close(conll_input)
    sentences#=,postaglists=#,parentlists#,deprellists
end

function test_eisner(sentences, parentlists, #=deprellists=#)
    n=length(sentences)
    #headlist=[]
    total=0
    correct=0
    for i=1:n
        s=length(sentences[i])
        total=total+s
        lambda_gold=zeros(s+1,s+1)
        for j=1:s
            if isa(parentlists[i][j], Int)
                lambda_gold[parentlists[i][j]+1,j+1]=1.0
            end
        end
        heads=eisner(lambda_gold)
        correct=correct+count(heads.==parentlists[i])
        #push!(headlist,heads)
    end
    #headlist

    100*correct/total
end

function test(conllfile)
    sentences, parentlists = read_conllfile(conllfile)
    test_eisner(sentences, parentlists)
end

path="/scratch/users/okirnap/ud-treebanks-v2.2/"

for directory in readdir(path)
    if directory[1:3]=="UD_"
        subdirectory=readdir(string(path,directory))
        for file in subdirectory
            if file[end-6:end] == ".conllu"
                println("Testing Eisner on: ",string(path,directory,"/",file))
                println(test(string(path,directory,"/",file)))
            end
        end
    end
end
