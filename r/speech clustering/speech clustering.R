# Toy example of the dynamic Tree Cut algorithms.
# Author: Peter Langfelder, Peter dot Langfelder at gmail dot com

# If necessary, modify the directory in the following statement to point to your working directory and
# execute the command. 

# setwd("C:/Documents and Settings/Work/TreeCut/Simulation");

source("supporting functions.R");


library(dynamicTreeCut);
library(moduleColor);
library(class);
library(clv);
library(stringdist);
library(fpc);


logDir = "E:/eclipse-log"

#
# load the data
#

# customer data
condition = "20191121_simulated_data_csshkputts_withsymbols_200 shopkeeper - tri stm - 1 wgt kw - mc2 - stopwords 1"

subUttData <- read.csv(file=paste("C:/Users/robovie/eclipse-workspace/DatabaseLearning/data/utterance vectors/", condition, " - utterance data.csv", sep=""), 
                    header=TRUE, sep=",")

numUtts <- length(subUttData$Utterance)

subDistMatrix <- matrix(scan(paste("C:/Users/robovie/eclipse-workspace/DatabaseLearning/data/utterance vectors/", condition, " - utterance cos dists.txt", sep="")),
                     nrow=numUtts, byrow=TRUE)





#
# cluster
#
sessionDir = paste(logDir, format(Sys.time(), "/%Y-%m-%d_%H-%M-%S_cameraCustomerSpeechClustering"), sep="")
dir.create(sessionDir)
saveDir <- paste(sessionDir, "/", sep="")


#subDistMatrix <- distMatrix
#subUttData <- uttData





#
# prepare data for clustering  
#
subUttData$uttIndices <- (1:numUtts)
subUttData$clusterIds <- rep("", times=numUtts)
subUttData$isJunk <- rep(0, times=numUtts)
subUttData$isRepresentative <- rep(0,numUtts)



#
# do the clustering
#
dissim <- as.dist(subDistMatrix)

dendro = hclust(dissim, method = "average");

rawClusters <- cutreeHybrid(dendro,
                            cutHeight = 0.99,
                            minClusterSize = 5,
                            deepSplit = 4,
                            pamStage = TRUE,
                            distM = subDistMatrix, 
                            maxPamDist = 0,
                            useMedoids=FALSE,
                            respectSmallClusters = FALSE,
                            pamRespectsDendro=FALSE,
                            verbose = 0)



#
# find the clusters that result from the tree cut
#
subUttData$clusterIds <- rawClusters$labels
uniqueClusters <- sort(unique(subUttData$clusterIds))
numClusters <- length(uniqueClusters)


#
# detect bad clusters
#

# add 1 so which(measurement$intracls.average > max(measurement$intracls.average) doesn't throw an error
clusterIdsPlus1 <- subUttData$clusterIds + 1

measurement <- cls.scatt.diss.mx(subDistMatrix, clusterIdsPlus1)

#junk.clusters <- which(measurement$intracls.average > max(measurement$intracls.average) * 0.80)
junk.clusters <- which(measurement$intracls.average > 0.355) #0.3285)

#junk.clusters <- c(junk.clusters, which(measurement$cluster.size < 5))
junk.clusters <- unique((junk.clusters))


# convert back to the original cluster ids
junk.clusters <- junk.clusters - 1


subUttData$isJunk <- as.integer(subUttData$clusterIds %in% junk.clusters)
numJunkUtts <- sum(subUttData$isJunk)

#
# find representative utterances for each cluster
#
repUtts <- sapply(uniqueClusters, TypicaSpeechFinder, subUttData$clusterIds, subDistMatrix)
subUttData$isRepresentative[(subUttData$uttIndices) %in% repUtts] <- 1



#
# write cluster metrics out to file
#
clusterSize <- measurement$cluster.size
intraComplete <- measurement$intracls.complete[1,]
intraAverage <- measurement$intracls.average[1,]
interComplete <- measurement$intercls.complete[1,]
interSingle <- measurement$intercls.single[1,]
interAverage <- measurement$intercls.average[1,]
interHausdorff <- measurement$intercls.hausdorff[1,]

# don't include the noise cluster when computing averages
aveClusterSize <- sum(tail(clusterSize, -1)) / (numClusters-1)
aveIntraComplete <- sum(tail(intraComplete, -1)) / (numClusters-1)
aveIntraAverage <- sum(tail(intraAverage, -1)) / (numClusters-1)
aveInterComplete <- sum(tail(interComplete, -1)) / (numClusters-1)
aveInterSingle <- sum(tail(interSingle, -1)) / (numClusters-1)
aveInterAverage <- sum(tail(interAverage, -1)) / (numClusters-1)
aveInterHausdorff <- sum(tail(interHausdorff, -1)) / (numClusters-1)

dunnIndex <- clv.Dunn(measurement, "average", "average")[1]
daviesBouldinIndex <- clv.Davies.Bouldin(measurement, "average", "average")[1]

numJunkClusters <- length(junk.clusters)


#
# write the info out to file
#

# aggregate stats

aggregateStats <- data.frame(numUtts,
                             numClusters,
                             aveClusterSize,
                             numJunkClusters,
                             aveIntraComplete,
                             aveIntraAverage,
                             aveInterComplete,
                             aveInterSingle,
                             aveInterAverage,
                             aveInterHausdorff,
                             dunnIndex,
                             daviesBouldinIndex)

write.csv(aggregateStats, file=paste(saveDir, "aggregate stats.csv", sep=""), row.names=FALSE)


# clusters

# for nice printing
Utterance.ID <- subUttData$Utterance.ID
Timestamp <- subUttData$Timestamp
Trial.ID <- subUttData$Trial.ID
Condition<- subUttData$Condition
Cluster.ID <- subUttData$clusterIds
Is.Representative <- subUttData$isRepresentative
Is.Junk <- subUttData$isJunk
Utterance <- subUttData$Utterance
Cluster.Size <- clusterSize[clusterIdsPlus1]
Intra.Complete <- intraComplete[clusterIdsPlus1]
Intra.Average <- intraAverage[clusterIdsPlus1]
Inter.Complete <- interComplete[clusterIdsPlus1]
Inter.Single <- interSingle[clusterIdsPlus1]
Inter.Average <- interAverage[clusterIdsPlus1]
Inter.Hausforff <- interHausdorff[clusterIdsPlus1]


clusterInfo <- data.frame(Utterance.ID,
                          Timestamp,
                          Trial.ID,
                          Condition,
                          Cluster.ID,
                          Is.Representative,
                          Is.Junk,
                          Utterance,
                          Cluster.Size,
                          Intra.Complete,
                          Intra.Average,
                          Inter.Complete,
                          Inter.Single,
                          Inter.Average,
                          Inter.Hausforff)

clusterInfo <- clusterInfo[order(Cluster.ID, Is.Representative),]

write.csv(clusterInfo, file=paste(saveDir, condition, "- speech_clusters.csv", sep=""), row.names=FALSE)


print(paste("           numUtts:", toString(numUtts), sep=""))
print(paste("   numUtts in good:", toString(numUtts - numJunkUtts), sep=""))
print(paste("       numClusters:", toString(numClusters), sep=""))
print(paste("   numJunkClusters:", toString(numJunkClusters), sep=""))
print(paste("    aveClusterSize:", toString(round(aveClusterSize, 2)), sep=""))
print(paste("  aveIntraComplete:", toString(round(aveIntraComplete, 2)), sep=""))
print(paste("   aveIntraAverage:", toString(round(aveIntraAverage, 2)), sep=""))
print(paste("  aveInterComplete:", toString(round(aveInterComplete, 2)), sep=""))
print(paste("    aveInterSingle:", toString(round(aveInterSingle, 2)), sep=""))
print(paste("   aveInterAverage:", toString(round(aveInterAverage, 2)), sep=""))
print(paste(" aveInterHausdorff:", toString(round(aveInterHausdorff, 2)), sep=""))
print(paste("         dunnIndex:", toString(round(dunnIndex, 2)), sep=""))
print(paste("daviesBouldinIndex:", toString(round(daviesBouldinIndex, 2)), sep=""))



# clusterColors = labels2colors(uniqueClusters)
# dendroDendrogram = as.dendrogram(dendro)
# dendroDendrogram <- color_branches(dend=dendroDendrogram, clusters=subUttData$clusterIds)
# #dendroDendrogram <- color_branches(dend=dendroDendrogram, clusters=subUttData$clusterIds)
# plot(dendroDendrogram)



# DetectedColors = NULL;
# DetectedColors = cbind(DetectedColors, labels2colors(subUttData$clusterIds));
# DetectedColors = cbind(DetectedColors, labels2colors(subUttData$clusterIds));
# 
# Methods = c("Speech Clusters 2", "Speech Clusters 1");
# 
# StandardCex = 1.7;
# width = 9;
# SizeWindow(width, 4);
# 
# layout(matrix(c(1,2), nrow = 2, ncol = 1), widths = 1, heights = c(0.8, 0.2));
# 
# 
# par(cex = 1.4);
# 
# par(mar=c(0, 6.5, 2, 0.2));
# 
# #cluster1colors = labels2colors(clusters1)
# #dendro_dendrogram = as.dendrogram(dendro)
# 
# #dendro_dendrogram <- color_branches(dend=dendro_dendrogram, col=cluster1colors, groupLabels=TRUE, clusters=clusterIds1Plus1)
# labels <- rep("", times=numUtts)
# plot(dendroDendrogram, labels = labels, main = "Toy example: clustering dendrogram and module colors", ylab = "Difference");
# 
# par(mar=c(0.2, 6.5, 0, 0.2));
# 
# hclustplotn(dendroDendrogram, DetectedColors, RowLabels = Methods, main="");




