require(mxnet)
require(readr)
library(magrittr)
library(stringr)
library(data.table)
library(text2vec)
library(jiebaR)
library(rvest)

spider=function(path){
  result=read_html(path) %>% 
    html_nodes("pre") %>% 
    html_text() %>%
    str_replace_all(pattern = "[0-9,a-z,A-Z]|―",replacement = "")
  return(result)
}

result=list()

for ( i in c(paste(0,1:9,sep=""),10:60)) {
  tryCatch({
    path=str_replace("http://www.shigeku.org/shiku/xs/yuguangzhong/0@.htm",pattern = "@",replacement =as.character(i))
    result[[i]]=spider(path)
  },error=function(e){""})
  
  setTxtProgressBar(txtProgressBar(min=0,max=1,style=3),value=as.numeric(str_extract(i,paste(60:1,collapse = "|")))/60)
}

text=paste(unlist(result),collapse = "@") %>% rep(30) %>% paste(collapse = "")

voc= segment(text,worker(symbol = T)) 

make.dict <- function(text, max.vocab=10000) {
  dic=list()
  voc_unique=unique(voc)
  index=1
  for (v in voc_unique){
    dic[[v]]=index
    index=index+1
  }
  return(dic)

  # text <- strsplit(text, '')
  # dic <- list()
  # idx <- 1
  # for (c in text[[1]]) {
  #   if (!(c %in% names(dic))) {
  #     dic[[c]] <- idx
  #     idx <- idx + 1
  #   }
  # }
  # if (length(dic) == max.vocab - 1)
  #   dic[["UNKNOWN"]] <- idx
  # cat(paste0("Total unique char: ", length(dic), "\n"))
  # return (dic)
}

batch.size = 64
seq.len = 3
num.hidden = 16
num.embed = 16
num.lstm.layer = 1
num.round = 1000
learning.rate= 0.6
wd=0.00001
clip_gradient=1
update.period = 1


make.data <- function(text, seq.len, max.vocab=10000, dic=NULL){
  
  
  if (is.null(dic))
    dic <- make.dict(text, max.vocab)
  lookup.table <- list()
  for (c in names(dic)) {
    idx <- dic[[c]]
    lookup.table[[idx]] <- c
  }
  
  #char.lst <- strsplit(text, '')[[1]]
  char.lst<- voc
  num.seq <- as.integer(length(char.lst) / seq.len)
  char.lst <- char.lst[1:(num.seq * seq.len)]
  data <- array(0, dim=c(seq.len, num.seq))
  idx <- 1
  for (i in 1:num.seq) {
    for (j in 1:seq.len) {
      if (char.lst[idx] %in% names(dic))
        data[j, i] <- dic[[ char.lst[idx] ]]-1
      else {
        data[j, i] <- dic[["UNKNOWN"]]-1
      }
      idx <- idx + 1
    }
  }
  return (list(data=data, dic=dic, lookup.table=lookup.table))
}

drop.tail <- function(X, batch.size) {
  shape <- dim(X)
  nstep <- as.integer(shape[2] / batch.size)
  return (X[, 1:(nstep * batch.size)])
}

get.label <- function(X) {
  label <- array(0, dim=dim(X))
  d <- dim(X)[1]
  w <- dim(X)[2]
  for (i in 0:(w-1)) {
    for (j in 1:d) {
      label[i*d+j] <- X[(i*d+j)%%(w*d)+1]
    }
  }
  return (label)
}


#text=paste(readLines("D://mxnet/input.txt"),collapse="\n")
# text=read_csv("E:/pathology/DFGDRaw.csv")$病理诊断 %>% 
#   paste0(collapse = "") %>%
#  str_replace_all(pattern = "[。,，,；,\\.,\\,,;,、,%,:,?,？,⑤,\\(]",replacement="")


ret <- make.data(text, seq.len=seq.len)

X <- ret$data
dic <- ret$dic
lookup.table <- ret$lookup.table

vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]

X.train.data <- X[, 1:as.integer(size * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(size * train.val.fraction))]
X.train.data <- drop.tail(X.train.data, batch.size)
X.val.data <- drop.tail(X.val.data, batch.size)

X.train.label <- get.label(X.train.data)
X.val.label <- get.label(X.val.data)

X.train <- list(data=X.train.data, label=X.train.label)
X.val <- list(data=X.val.data, label=X.val.label)

logger <- mx.metric.logger$new()

model <- mx.lstm(X.train, X.val,
                 ctx=mx.cpu(),
                 num.round=num.round,
                 update.period=update.period,
                 num.lstm.layer=num.lstm.layer,
                 seq.len=seq.len,
                 num.hidden=num.hidden,
                 num.embed=num.embed,
                 num.label=vocab,
                 batch.size=batch.size,
                 input.size=vocab,
                 initializer=mx.init.uniform(0.1),
                 learning.rate=learning.rate,
                 wd=wd,
                 clip_gradient=clip_gradient,
                 epoch.end.callback =mx.callback.save.checkpoint("YGZ",num.round))


cdf <- function(weights) {
  total <- sum(weights)
  result <- c()
  cumsum <- 0
  for (w in weights) {
    cumsum <- cumsum+w
    result <- c(result, cumsum / total)
  }
  return (result)
}

search.val <- function(cdf, x) {
  l <- 1
  r <- length(cdf)
  while (l <= r) {
    m <- as.integer((l+r)/2)
    if (cdf[m] < x) {
      l <- m+1
    } else {
      r <- m-1
    }
  }
  return (l)
}
choice <- function(weights) {
  cdf.vals <- cdf(as.array(weights))
  x <- runif(1)
  idx <- search.val(cdf.vals, x)
  return (idx)
}

make.output <- function(prob, sample=FALSE) {
  if (!sample) {
    idx <- which.max(as.array(prob))
  }
  else {
    idx <- choice(prob)
  }
  return (idx)
  
}

infer.model <- mx.lstm.inference(num.lstm.layer=num.lstm.layer,
                                 input.size=vocab,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=vocab,
                                 arg.params=model$arg.params,
                                 ctx=mx.cpu())


start <- '我'
seq.len <- 50
random.sample <- TRUE

last.id <- dic[[start]]
out <- "我"
for (i in (1:(seq.len-1))) {
  input <- c(last.id-1)
  ret <- mx.lstm.forward(infer.model, input, FALSE)
  infer.model <- ret$model
  prob <- ret$prob
  last.id <- make.output(prob, random.sample)
  out <- paste0(out, lookup.table[[last.id]])
}
cat (paste0(out, "\n"))

