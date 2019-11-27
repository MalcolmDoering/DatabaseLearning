

TypicaSpeechFinder <- function(clustId, clusters, distMatrix) {
  
  # find the utterance that is the closest on average to all the other utterances in this cluster
  uttIndices <- which(clusters == clustId)
  
  #print(clustId)
  #print(uttIndices)
  
  minAveDist <- ""
  bestUttIndex <- ""
  
  for (i in uttIndices) {
    aveDist <- 0
    
    for (j in uttIndices) {
      aveDist = aveDist + distMatrix[i,j]
    }
    
    aveDist = aveDist / length(uttIndices)
    
    if (minAveDist == "" || aveDist < minAveDist) {
      minAveDist <- aveDist
      bestUttIndex <- i
    }
    
  }
  
  #print(clustId)
  #print(bestUttIndex)
  #print("")
  
  return(bestUttIndex)
}


#--------------------------------------------------------------------------
#
# SizeWindow
#
#--------------------------------------------------------------------------
# if the current device isn't of the required dimensions, close it and open a new one.

SizeWindow = function(width, height)
{
  din = par("din");
  if ( (din[1]!=width) | (din[2]!=height) )
  {
    dev.off();
    X11(width = width, height=height);
  }
}



if (exists("hclustplotn")) rm(hclustplotn);
hclustplotn=function(hier1, Color, RowLabels=NULL, cex.RowLabels = 0.9, ...) 
{
  options(stringsAsFactors=FALSE);
  if (length(hier1$order) != dim(Color)[[1]] ) 
  { 
    stop("ERROR: length of color vector not compatible with no. of objects in the hierarchical tree");
  } else {
    No.Sets = dim(Color)[[2]];
    C = Color[hier1$order, ]; 
    step = 1/dim(Color)[[1]];
    ystep = 1/No.Sets;
    barplot(height=1, col = "white", border=F,space=0, axes=F, ...)
    for (j in 1:No.Sets)
    {
      ind = (1:(dim(C)[1]));
      xl = (ind-1) * step; xr = ind * step; 
      yb = rep(ystep*(j-1), dim(C)[1]); yt = rep(ystep*j, dim(C)[1]);
      rect(xl, yb, xr, yt, col = as.character(C[,j]), border = as.character(C[,j]));
      if (is.null(RowLabels))
      {
        text(as.character(j), pos=2, x=0, y=ystep*(j-0.5), cex=cex.RowLabels, xpd = TRUE);
      } else {
        text(RowLabels[j], pos=2, x=0, y=ystep*(j-0.5), cex=cex.RowLabels, xpd = TRUE);
      }
    }
    for (j in 1:No.Sets) lines(x=c(0,1), y=c(ystep*j,ystep*j));
  }
}