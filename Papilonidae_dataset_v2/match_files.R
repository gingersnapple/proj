
# Load all files

metadata = read.table("./Papilonidae_dataset_v1/Papilonidae_metadata.txt",sep=",",header=T)
landmarks = read.table("./Papilonidae_dataset_v1/Papilionidae_landmarks.txt",sep="\t",header=F)
aligned =  read.table("./Papilonidae_dataset_v1/Papilionidae_aligned.txt",sep="\t",header=F)
tree = read.tree("./Papilonidae_dataset_v1/Papilionidae_tree.txt")

length(intersect(landmarks[,1],metadata$match_landmarks))

length(intersect(tree$tip.label,metadata$match_tree))







##########################
###########################
# Convert a single subgroup (from split data frame) into a 2D slice
convert_to_slice <- function(df_subgroup) {
  x_coords <- as.numeric(df_subgroup[df_subgroup[,2] == "x-coordinates", -c(1,2)])
  y_coords <- as.numeric(df_subgroup[df_subgroup[,2] == "y-coordinates", -c(1,2)])
  
  return(cbind(unlist(x_coords), unlist(y_coords)))
}

# Split dataframe by "Name" and convert each subgroup back to a 2D slice
list_of_2d_slices <- lapply(split(landmarks, landmarks[,1]), convert_to_slice)

# Convert the list of 2D slices into a 3D array
array3D_reconstructed <- abind::abind(list_of_2d_slices, along=3)

# View the reconstructed 3D array
print(array3D_reconstructed)
###################
###################
make_slide_curves  <- function(landmarks,p){
  
  slidingcurves <- matrix(0,ncol=3,nrow=p)
  slidingcurves[,1] <- 0:(p-1) ; slidingcurves[,2] <- 1:(p) ;slidingcurves[,3] <- 2:(p+1)
  slidingcurves[1,1] <- p ; slidingcurves[p,3] <- 1
  return(slidingcurves)
}

##############################
##############################


gdf <- geomorph.data.frame("landmarks"=array3D_reconstructed,"species"=metadata$species)
Y.gpa1 <- gpagen(gdf$landmarks,print.progress = T,ProcD = F) # GPA-alignment


p = dim(gdf[[1]])[1]
slidingcurves = make_slide_curves(gfd$landmarks,p=dim(gdf[[1]])[1])
Y.gpa <- gpagen(gdf$landmarks,curves=slidingcurves[3:(p-3),],print.progress = T,ProcD = T) # GPA-alignment

par(mfrow=c(1,2))
plot(Y.gpa1)
plot(Y.gpa)


Y.gpa3 <- gpagen(Y.gpa$coords,print.progress = T,ProcD = F) # GPA-alignment
plot(Y.gpa3)

##############################
##############################
# 2d
# data(plethodon)
# Y.gpa <- gpagen(plethodon$land)
# pleth.pca <- gm.prcomp(Y.gpa$coords)
# pleth.pca.plot <- plot(pleth.pca)
# picknplot.shape(pleth.pca.plot)
# May change arguments for plotRefToTarget
# picknplot.shape(plot(pleth.pca), method = "points", mag = 3,
# links=plethodon$links)


mypca = gm.prcomp(Y.gpa$coords)

for(i in 1:2){
  if(i == 1)mycoords = Y.gpa else mypca = mycoords = Y.gpa1
  mypca = gm.prcomp(mycoords$coords)
  M <- mshape(mycoords$coords)
  par(mfrow=c(1,1))
  plot(mypca,col=metadata$swallowtail+1)
  legend("topleft", legend=c("no tail", "swallow tail"),
         col=c(1,2),pch=c(1,1,1) ,cex=1.2)
  PC <- mypca$x[,1:2]
  preds <- shape.predictor(mycoords$coords, x= PC, Intercept = FALSE, 
                           pred1 = c(min(PC[,1]),max(PC[,2])), 
                           pred2 = c(max(PC[,1]),max(PC[,2])), 
                           pred3 = c(min(PC[,1]),min(PC[,2])),
                           pred4 = c(max(PC[,1]),min(PC[,2])))
  par(mfrow=c(2,2))
  plotRefToTarget(M, preds$pred1)
  plotRefToTarget(M, preds$pred2)
  plotRefToTarget(M, preds$pred3)
  plotRefToTarget(M, preds$pred4)
}


###########################
################################
###########################


Y.gpa4 <- gpagen(gdf$landmarks[26:50,,],print.progress = T,ProcD = F) # GPA-alignment
plot(Y.gpa4)



mypca = gm.prcomp(Y.gpa4$coords)
par(mfrow=c(1,1))
plot(mypca,col=metadata$swallowtail+1)
legend("topleft", legend=c("no tail", "swallow tail"),
       col=c(1,2),pch=c(1,1,1) ,cex=1.2)
PC <- mypca$x[,1:2]
preds <- shape.predictor(Y.gpa4$coords, x= PC, Intercept = FALSE, 
                         pred1 = c(min(PC[,1]),max(PC[,2])), 
                         pred2 = c(max(PC[,1]),max(PC[,2])), 
                         pred3 = c(min(PC[,1]),min(PC[,2])),
                         pred4 = c(max(PC[,1]),min(PC[,2])))
par(mfrow=c(2,2))
M <- mshape(Y.gpa4$coords)
plotRefToTarget(M, preds$pred1)
plotRefToTarget(M, preds$pred2)
plotRefToTarget(M, preds$pred3)
plotRefToTarget(M, preds$pred4)



Y.gpa4 <- gpagen(gdf$landmarks[1:25,,],print.progress = T,ProcD = F) # GPA-alignment
plot(Y.gpa4)



mypca = gm.prcomp(Y.gpa4$coords)
par(mfrow=c(1,1))
plot(mypca,col=metadata$swallowtail+1)
legend("topleft", legend=c("no tail", "swallow tail"),
       col=c(1,2),pch=c(1,1,1) ,cex=1.2)
PC <- mypca$x[,1:2]
preds <- shape.predictor(Y.gpa4$coords, x= PC, Intercept = FALSE, 
                         pred1 = c(min(PC[,1]),max(PC[,2])), 
                         pred2 = c(max(PC[,1]),max(PC[,2])), 
                         pred3 = c(min(PC[,1]),min(PC[,2])),
                         pred4 = c(max(PC[,1]),min(PC[,2])))
par(mfrow=c(2,2))
M <- mshape(Y.gpa4$coords)
plotRefToTarget(M, preds$pred1)
plotRefToTarget(M, preds$pred2)
plotRefToTarget(M, preds$pred3)
plotRefToTarget(M, preds$pred4)


############################################################
###########################################################

colors <- c("red", "blue", "green", "purple")

# Define the range list
ranges <- list(1:25, 26:50, 51:75, 76:100)


idx = (1:length(metadata$id))[!duplicated(metadata$species)]

par(mfrow=c(2,3))
for(i in idx){
  points_my = gdf$landmarks[,,i]
  plot(points_my,main=metadata$species[i],xlab="",ylab="")
  
  for (j in 1:4) {
    points_temp <- points_my[ranges[[j]], ]
    points(points_temp[,1],points_temp[,2],col=j)
    lines(points_temp[,1],points_temp[,2],col=j)
    
  }
  points_my = Y.gpa1$coords[,,i]
  plot(points_my,main="aligned",xlab="",ylab="")
  
  for (j in 1:4) {
    points_temp <- points_my[ranges[[j]], ]
    points(points_temp[,1],points_temp[,2],col=j)
    lines(points_temp[,1],points_temp[,2],col=j)
  }
  
  points_my = Y.gpa$coords[,,i]
  plot(points_my,main="aligned with semilandmarks",xlab="",ylab="")
  
  for (j in 1:4) {
    points_temp <- points_my[ranges[[j]], ]
    points(points_temp[,1],points_temp[,2],col=j)
    lines(points_temp[,1],points_temp[,2],col=j)
  }
  
  
}













############################################################
###########################################################
############################################################
###########################################################
# Estimate rates

Y.gpa_fore <- gpagen(gdf$landmarks[1:25,,],print.progress = T,ProcD = F) # GPA-alignment


dimnames(Y.gpa_fore$coords)[[3]] <- metadata$match_tree

my_gp <- as.factor(as.numeric(metadata$swallowtail[!duplicated(metadata$match_tree)]))
names(my_gp) <-  metadata$match_tree[!duplicated(metadata$match_tree)]

compare.evol.rates(Y.gpa_fore$coords,phy=tree,gp=my_gp)



Y.gpa_hind <- gpagen(gdf$landmarks[26:50,,],print.progress = T,ProcD = F) # GPA-alignment

dimnames(Y.gpa_hind$coords)[[3]] <- metadata$match_tree


my_gp <- as.factor(as.numeric(metadata$swallowtail[!duplicated(metadata$match_tree)]))
names(my_gp) <-  metadata$match_tree[!duplicated(metadata$match_tree)]

compare.evol.rates(Y.gpa_hind$coords,phy=tree,gp=my_gp)



#####################

Y.gpa_hind <- gpagen(gdf$landmarks[26:50,,],print.progress = T,ProcD = F) # GPA-alignment


physignal(A=Y.gpa_hind$coords[,,1:49],phy=tree,iter=99)



data(plethspecies)
Y.gpa<-gpagen(plethspecies$land) #GPA-alignment
#Test for phylogenetic signal in shape
PS.shape <- physignal(A=Y.gpa$coords,phy=plethspecies$phy,iter=999)
summary(PS.shape)
plot(PS.shape)
plot(PS.shape$PACA, phylo = TRUE)
PS.shape$K.by.p # Phylogenetic signal profile
#Test for phylogenetic signal in size
PS.size <- physignal(A=Y.gpa$Csize,phy=plethspecies$phy,iter=999)
summary(PS.size)
plot(PS.size)


##########################
morphol.disparity(Y.gpa$coords~as.factor(metadata$swallowtail),)


coords.subset




data(pupfish)
group <- factor(paste(pupfish$Pop, pupfish$Sex))
levels(group)
new.coords <- coords.subset(A = pupfish$coords, group = group)
names(new.coords) # see the list levels
# group shape means
lapply(new.coords, mshape)












