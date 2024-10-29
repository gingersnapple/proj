 
 21/09-2023

Landmarks for papilionidae, matched with tree from Kawahara 2023

The landmarks are aligned procrusted with "gpagen" from the R package "Geomorph" 
The code there is executed is:  
Y.gpa <- gpagen(df1$landmarks,print.progress = T,ProcD = F)

The df1$landmarks, corresponds to the entire 100 landmarks from all 4 wings. 

For matching the different files, and get the individual wing. See example documents made in R and Python


For each butterfly there is 100 landmarks from the wing outlie. 
The landmarks can be divided into 4 subset for each wing
 1:25 ; Left Forewing 
26:50 ; Right Hindwing
51:75 ; Left Hindwing
76:100; Right Forewing 


