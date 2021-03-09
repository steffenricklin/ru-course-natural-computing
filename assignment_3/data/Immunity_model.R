#import the anomaly scores 
library(pROC)
options(digits = 20)

english_output <- as.numeric(readLines("outenglish.txt"))
tagalog_output <-as.numeric(readLines("outtagalog.txt"))
anomaly_score <- c(english_output,tagalog_output)
label<- c(rep("english", length(english_output)),rep("tagalog",length(tagalog_output)))
r <- c(rep(4,length(label)))
df <- data.frame(anomaly_score,label,r)

roc_list = c()
for (i in seq(1,13)){
  english_output <- as.numeric(readLines(paste("englishr",as.character(i),".txt",sep="")))
  tagalog_output <-as.numeric(readLines(paste("tagalogr",as.character(i),".txt",sep="")))
  anomaly_score <- c(english_output,tagalog_output)
  label<- c(rep("english", length(english_output)),rep("tagalog",length(tagalog_output)))
  r <- c(rep(i,length(label)))
  if (i != 1){
   df <- rbind(df,data.frame(anomaly_score,label,r)) 
  }
  else{
    df <-data.frame(anomaly_score,label,r)
  }
  roc_list  = cbind(roc_list,roc( df$label[which(df$r == i)],  df$anomaly_score[which(df$r == i)], levels = c("english","tagalog"))
$auc)
}

roc_list[4]
#AUC for r = 4: 0.79160971386914525

roc_list[1]
roc_list[9]
#AUC for r=1 : 0.54353471842536916
#and r=9 : 0.51209677419354838
plot.roc(roc( df$label[which(df$r == 1)],  df$anomaly_score[which(df$r == 1)], levels = c("english","tagalog")))
plot(roc( df$label[which(df$r == 9)],  df$anomaly_score[which(df$r == 9)], levels = c("english","tagalog")))
# ROC curve
# the x-axis showing  specificity (= false positive fraction = FP/(FP+TN))
# the y-axis showing sensitivity (= true positive fraction = TP/(TP+FN))
# As we can see, the ROC curve is slightly above the 0.5 line for both r values
# axcept they are above it a the opposite places. This was to be expected, since ....
#TO DO: wxplain the roc curve



plot(seq(1,10),roc_list)
roc_list[3]
#The best AUC value is for r=3 with 0.83112356478950244


