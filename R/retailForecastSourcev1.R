####COPYRIGHT EVERLYTICS 2013
####FOR CLIENT USE ONLY


pennington_harvest <- function(fileName='forecast',dataSource="harvest",writeDataSource="harvest temp",column="pos_qty",inputTable="harvest_temp.dbo.pennington_data",outputTable="Penn_Store_Forecast",user="everlytics",pass="ravenlikeawritingdesk",skuLevel=TRUE,weeksOut=52)
  
{
  
  options(warn=-1)
  
  # harvest stuff here
  library('forecast') #loads forecast package
  library(RODBC) #loads ODBC package
  library(zoo)
  
  #declare connection to database object - DSN is dependent on what it is named on machine
  
  db <- odbcConnect(dataSource,uid=user,pwd=pass)
  
  #create a vector of unique storeIDs to loop through
  
  
  if(skuLevel)
  {
    storeVector <- sqlQuery(db, paste("select distinct cast(store_nbr as varchar) + ' ' + cast(item_nbr as varchar) as store_nbr from ", inputTable))
  }
  else
  {
    storeVector <- sqlQuery(db, paste("select distinct store_nbr from ",inputTable))
  }
  #initialize looping variables
  i <- 1
  N <- 10
  
  #nrow(storeVector)
  
  #loop through the stores
  results<-data.frame()
  #colnames(results) <- c("date","store","forecast","lower80","upper80","lower95","upper95","actual")
  while(i <= N) 
  {
    
    #query for one store at a time  
    if(skuLevel)
    {
      y<-unlist(strsplit(as.character(storeVector[i,1])," "))
      stor <- as.integer(y[1])
      item <- as.integer(y[2])
      q <-sqlQuery(db, paste("select week, sum(qty) as qty from (select case when right(wm_week,2)='53' then
                             left(wm_week,4) + '52' else wm_week end as week, ", column," as qty 
                             from   
                             ", inputTable, "
                             where 
                             store_nbr = ", stor,
                             " and item_nbr = ", item," and wm_week < year(getdate())*100 + DATEPART(wk, GETDATE())
      ) a
                             group by week order by week
                             
                             "))
    }
    else
    {
      q <-sqlQuery(db, paste("select week, sum(qty) as qty from (select case when right(wm_week,2)='53' then
                             left(wm_week,4) + '52' else wm_week end as week, ", column," as qty 
                             from   
                             ", inputTable ,"
                             where 
                             store_nbr = ", storeVector[i,1]," and wm_week < year(getdate())*100 + DATEPART(wk, GETDATE() )) a
                             group by week order by week
                             "))
    }
    
    
    #error handling for stores with less than 2 years worth of data
    
    if(nrow(q)<=104)
    {
      i <- i+1
    } 
    else
    {
      
      qtyChk <- aggregate(q$qty,by=list(year=substr(q$week,1,4)), FUN=sum)
      
      qtyChk<-subset(qtyChk, year == substr(Sys.Date(),1,4) | x != 0)
      q<-subset(q, substr(q$week,1,4) %in% qtyChk$year)
      x<-ts(zoo(q$qty,as.Date(paste(substr(q$week,1,4),'-01-01',sep='')) + (as.numeric(substr(q$week,5,7))*7-1)),freq=52)
      
      #Arima Automtic Fit based on AIC and below parameters
      
      fc12 <- auto.arima(x, max.p=3,max.q=3, D=1, d=0, max.Q=0, max.P=0,   		
                         max.order=6,start.p=0, start.q=0, start.P=0, start.Q=1, stationary=FALSE, 
                         seasonal=TRUE, ic=("aic"), stepwise=TRUE, 
                         parallel=TRUE,approximation=TRUE)
      
      #forecast out a year
      
      fc11 <- forecast(fc12, h=weeksOut)
      
      
      datez<-as.integer(q$week)
      maxDate<-max(datez)
      futureWeeks<-as.integer(format(seq(as.Date("2013-01-01"),as.Date("2050-01-01"),by="week"), "%Y%U"))
      futureWeeks<-subset(futureWeeks,futureWeeks>maxDate)
      future<-futureWeeks[1:weeksOut]
      
      #create a data table of a year's worth of forecast data in the future
      
      aa<-summary(fc11,digits = 2)
      #combine Data together
      if(skuLevel)
      {
        ra<-data.frame(datez,stor,item,as.numeric(fc11$fitted),NA,NA,NA,NA,q$qty)
        a<-data.frame(future,stor,item,aa,NA)
        colnames(ra) <- c("Date","Store_Nbr","Item_Nbr","Forecast","Lower_80","Upper_80","Lower_95","Upper_95","Actual")
        colnames(a) <- c("Date","Store_Nbr","Item_Nbr","Forecast","Lower_80","Upper_80","Lower_95","Upper_95","Actual")
        
      }
      else
      {
        ra<-data.frame(as.character(datez),storeVector[i,1],as.numeric(fc11$fitted),NA,NA,NA,NA,q$qty)
        a<-data.frame(as.character(future),storeVector[i,1],aa,NA)
        colnames(ra) <- c("Date","Store_Nbr","Forecast","Lower_80","Upper_80","Lower_95","Upper_95","Actual")
        colnames(a) <- c("Date","Store_Nbr","Forecast","Lower_80","Upper_80","Lower_95","Upper_95","Actual")
      }
      a<-rbind(ra,a)
      
      
      results<-rbind(results,a)
      
      ##INSERT RODBC HERE
      
      
      #next iteration of the loop		
      
      i <- i+1
    } #close ELSE
    } #close WHILE
  write.table(results,file=paste(fileName,".csv",sep=""), sep=",", na="",row.names=FALSE)
  #  db2 <- odbcConnect(writeDataSource,uid=user,pwd=pass)
  # sqlUpdate(db2,results,tablename=outputTable)
    }