# CHANGE WITH YOUR DIRECTORY PATH
setwd(dir="path/")

#############################################
### proba collision - 0.1 - with Kronecker
#############################################
pdf(paste0("collision256-64_Kronecker.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("collision256-64.csv",sep=",",header=TRUE)
matplot(angular[,'distance'],angular[,-1],type="b",pch="+", xlim = c(0.1, sqrt(2)),
        ylim=c(0.01,1.0),log = "y", lty=1,
        axes=FALSE,ylab='Collision probability',
        xlab='Distance',col=1:7,
        main="Collision probabilities with cross-polytope LSH")

axis(2)
#axis(2, at = c(-4, -3, -2, -3, 0),labels=expression(10^-4, 10^-3, 10^-2, 10^-1, 10^0))
axis(1,at = seq(0.1,1.4,by=0.1),labels=expression(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, sqrt(2)))

text(x = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, sqrt(2)), y=par("usr")[3]-0.0006, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, sqrt(2)))


abline(v=seq(0.1,1.4,by=0.1),col='darkgray',lty=2,lwd=1)
#abline(h=seq(0,1.0,by=0.10),col='darkgray',lty=2,lwd=1)
abline(h=10^seq(-4,0, by = 1),col='darkgray',lty=2,lwd=1)

legend("bottomleft",legend = c(expression(G),
                               expression(G[circ]*K[2]*K[1]),
                               expression(G[Toeplitz]*D[2]*H*D[1]),
                               expression(G[skew-circ]*D[2]*H*D[1]),
                               expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                               expression(H*D[3]*H*D[2]*H*D[1])),
       col = 1:7,
       lwd=4,bg="white")
dev.off()



#############################################
### proba collision - 0.1 - with Kronecker + zoom 1-7
#############################################

pdf(paste0("collision256-64_Kronecker_zoom.pdf"),
    width=14,height=12,
    pointsize=32)

angular <- read.table("collision256-64.csv",sep=",",header=TRUE)

x = angular[,'distance']
y = angular[,-1]
limit = 4
x = x[11:14]
y = y[11:14,]

matplot(x, y ,type="b",pch="+", xlim = c(1.1, sqrt(2)),
        ylim=c(0.01, 0.10),log = "y", lty=1,
        axes=FALSE,ylab='Collision probability',
        xlab='Distance',col=1:7)


axis(2)

#axis(2, at = c(-4, -3, -2, -3, 0),labels=expression(10^-4, 10^-3, 10^-2, 10^-1, 10^0))
axis(1,at = seq(1.1,1.4,by=0.1),labels=expression(1.1, 1.2, 1.3, 1.4))

text(x = c(1.1, 1.2, 1.3, sqrt(2)), y=par("usr")[3]-0.0006, 
     srt = 0, pos = 1, xpd = TRUE,
     labels = expression(1.1, 1.2, 1.3, sqrt(2)))


abline(v=seq(1.0,sqrt(2),by=0.1),col='darkgray',lty=2,lwd=1)
#abline(h=seq(0,1.0,by=0.10),col='darkgray',lty=2,lwd=1)
abline(h=seq(0.01, 0.1,by=0.01),col='darkgray',lty=2,lwd=1)

legend("bottomleft",legend = c(expression(G),
                               expression(G[circ]*K[2]*K[1]),
                               expression(G[Toeplitz]*D[2]*H*D[1]),
                               expression(G[skew-circ]*D[2]*H*D[1]),
                               expression(HD[list(g[1],g[2],...,g[n])]*H*D[2]*H*D[1]),
                               expression(H*D[3]*H*D[2]*H*D[1])),
       col = 1:7,
       lwd=4.0,bg="white", cex = 0.9)
dev.off()



