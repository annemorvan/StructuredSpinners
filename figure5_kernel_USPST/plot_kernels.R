setwd(dir="path")


#############################################
### angular kernel
#############################################
pdf(paste0("angularkernel_G50C.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("angularkernel_G50C.csv",sep=",",header=TRUE)
matplot(2^angular[,'n'],angular[,-1],type="b",pch="+",
        ylim=c(0.01,0.14),log="x",lty=1,
        axes=FALSE,ylab='Gram matrix reconstruction error',
        xlab='Number of random features',col=1:7,
        main="Gram matrix reconstruction error\nG50C dataset for the angular kernel", line = 2.0)
axis(2)
axis(1,at = 2^seq(6,11,by=1),labels=FALSE)

text(x = 2^seq(6,11,by=1), y=par("usr")[3] - 0.0009, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(2^6,2^7,2^8,2^9,2^10,2^11))


abline(v=2^seq(6,11,by=1),col='darkgray',lty=2,lwd=1)
abline(h=seq(0.01,0.14,by=0.01),col='darkgray',lty=2,lwd=1)

legend("topright",legend = c(expression(G),
                             expression(G[circ]*K[2]*K[1]),
                             expression(G[Toeplitz]*D[2]*H*D[1]),
                             expression(G[skew-circ]*D[2]*H*D[1]),
                             expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                             expression(H*D[3]*H*D[2]*H*D[1])),
       col = 1:7,
       lwd=4,bg="white")
dev.off()



#############################################
### gaussian kernel
#############################################

pdf(paste0("gaussiankernel_G50C.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("gaussiankernel.csv",sep=",",header=TRUE)
matplot(2^angular[,'n'],angular[,-1],type="b",pch="+",
        ylim=c(0.007,0.05),log="x",lty=1,
        axes=FALSE,ylab='Gram matrix reconstruction error',
        xlab='Number of random features',col=1:7,
        main="Gram matrix reconstruction error\nG50C dataset for the Gaussian kernel", line = 2.0)
#main="Gram matrix reconstruction error\nG50C dataset for the Gaussian kernel", par = c(mar = c(5,7,4,2) + 0.1))
axis(2)
axis(1,at = 2^seq(6,11,by=1),labels=FALSE)

text(x = 2^seq(6,11,by=1), y=par("usr")[3] - 0.0008, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(2^6,2^7,2^8,2^9,2^10,2^11))


abline(v=2^seq(6,11,by=1),col='darkgray',lty=2,lwd=1)
abline(h=seq(0.000,0.05,by=0.005),col='darkgray',lty=2,lwd=1)

legend("topright",legend = c(expression(G),
                             expression(G[circ]*K[2]*K[1]),
                             expression(G[Toeplitz]*D[2]*H*D[1]),
                             expression(G[skew-circ]*D[2]*H*D[1]),
                             expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                             expression(H*D[3]*H*D[2]*H*D[1])
),
col = 1:7,
lwd=4,bg="white")
dev.off()


#############################################
### gaussian kernel - USPST
#############################################

pdf(paste0("gaussiankernel_USPST.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("gaussiankernel_USPST.csv",sep=",",header=TRUE)
matplot(2^angular[,'n'],angular[,-1],type="b",pch="+",
        ylim=c(0,0.12),log="x",lty=1,
        axes=FALSE,ylab='Gram matrix reconstruction error',
        xlab='Number of random features',col=1:7,
        main="Gram matrix reconstruction error \nUSPST dataset for the Gaussian kernel", line = 2.0)
axis(2)
axis(1,at = 2^seq(8,13,by=1),labels=FALSE)

text(x = 2^seq(8,13,by=1), y=par("usr")[3]-0.0006, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(2^8,2^9,2^10,2^11,2^12,2^13))


abline(v=2^seq(8,13,by=1),col='darkgray',lty=2,lwd=1)
abline(h=seq(0,0.12,by=0.02),col='darkgray',lty=2,lwd=1)

legend("topright",legend = c(expression(G),
                             expression(G[circ]*K[2]*K[1]),
                             expression(G[Toeplitz]*D[2]*H*D[1]),
                             expression(G[skew-circ]*D[2]*H*D[1]),
                             expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                             expression(H*D[3]*H*D[2]*H*D[1])
),
col = 1:7,
lwd=4,bg="white", cex = 0.9)
dev.off()



#############################################
### angular kernel - USPST 
#############################################

pdf(paste0("angularkernel_USPST.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("angularkernel_USPST.csv",sep=",",header=TRUE)
matplot(2^angular[,'n'],angular[,-1],type="b",pch="+",
        ylim=c(0.005,0.05),log="x",lty=1,
        axes=FALSE,ylab='Gram matrix reconstruction error',
        xlab='Number of random features',col=1:7,
        main="Gram matrix reconstruction error \nUSPST dataset for the angular kernel")
axis(2)
axis(1,at = 2^seq(8,13,by=1),labels=FALSE)

text(x = 2^seq(8,13,by=1), y=par("usr")[3]-0.0006, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(2^8,2^9,2^10,2^11,2^12,2^13))


abline(v=2^seq(8,13,by=1),col='darkgray',lty=2,lwd=1)
abline(h=seq(0.005,0.05,by=0.005),col='darkgray',lty=2,lwd=1)

legend("topright",legend = c(expression(G),
                             expression(G[circ]*K[2]*K[1]),
                             expression(G[Toeplitz]*D[2]*H*D[1]),
                             expression(G[skew-circ]*D[2]*H*D[1]),
                             expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                             expression(H*D[3]*H*D[2]*H*D[1])
),
col = 1:7,
lwd=4,bg="white", cex = 0.9)
dev.off()
