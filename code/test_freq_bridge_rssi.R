
# 내부수집정보


# packages load
# install.packages("needs")
library(needs)
needs(dplyr,tibble,lubridate,ggplot2,tidyr,readr,gridExtra)



# data load
data = read.csv("../data/500pois_collected_manytimes_reEncoded.csv",encoding="iso-8859-1")

data <- data %>% select(-X)

data$date <- substr(data$date_utc,1,10) 
data$hr <- substr(data$date_utc,12,13) %>% as.numeric()
data$mn <- substr(data$date_utc,15,16) %>% as.numeric()
data$se <- substr(data$date_utc,18,19) %>% as.numeric()
data <- data %>% select(-date_utc)
data <- data %>% select(date,hr,mn,se,pid,wifi,floor)
data$wifi <- data$wifi %>% paste("\t")



### 내부 수집된 매장 번호 개수

# table(data$pid)[table(data$pid) == 1]
# 470702 470968 470981 473412 473473
# 내부 수집이 한번 된 것 제외

data <- data %>% subset(!pid %in% (table(data$pid)[table(data$pid) == 1] %>% dimnames %>% .[[1]] %>% as.numeric))

table(data$pid) %>% length
# 497개

save_bridge <- c()

### 매장 선택
for (i in 1:length(data$pid %>% table %>% dimnames %>% .[[1]])){
nowpid <- data$pid %>% table %>% dimnames %>% .[[1]] %>% .[i]
data1 <- data %>% subset(pid == nowpid)


## EDA


new_data_test <- data1$wifi %>%
  strsplit("\t") %>%
  lapply(function(x){matrix(x,ncol=5,byrow=TRUE)[,c(1:4)]})

# wifi new table list
new_data <- data1$wifi %>%
  strsplit("\t") %>%
  lapply(function(x){matrix(x,ncol=5,byrow=TRUE)[,c(1,3,4)] %>% as.tibble(ncol = 3)})



# remove wifi information if there is only one
remove_list <- c()
j=1
for(i in 1:length(new_data)){
  if(new_data[[i]] %>% length == 1){
    remove_list[j] <- i
    j=j+1
  }
}

new_data[remove_list] <- NULL



# new format of wifi information 
for (i in 1:length(new_data)){
  new_data[[i]] <- data1 %>% .$date %>% .[i] %>% ymd %>% rep(nrow(new_data[[i]])) %>%
    cbind(index = i,
          new_data[[i]])
  
  colnames(new_data[[i]]) <- c("date","index","bssid","rssi","frequency")
}



# list to table
total_data <- new_data[[1]]
for(i in 2:length(new_data)){
  total_data <- rbind(total_data,new_data[[i]])
}
total_data$bssid <- as.factor(total_data$bssid)
total_data$rssi <- as.numeric(total_data$rssi)
total_data$frequency <- as.numeric(total_data$frequency)




total_data <- total_data %>% group_by(date, index, bssid) %>% summarise(rssi = mean(rssi), frequency = mean(frequency))



# pid가 변할 때 마다 와이파이나 주파수가 영향을 받을까? 동일한 환경이라고 할 수 있나? t.test 같은 것을 할 수 있을까?

# 상관도 테스트
spread_data <- total_data %>% select(index, bssid, frequency) %>% spread(bssid, frequency)
spread_data <- spread_data[,-c(1,2)] %>% t %>% as.data.frame
colnames(spread_data) <- c("fir","sec","thi","fou")

spread_data1 <- total_data %>% select(index, bssid, rssi) %>% spread(bssid, rssi)
spread_data1 <- spread_data1[,-c(1,2)] %>% t %>% as.data.frame 
colnames(spread_data1) <- c("fir","sec","thi","fou")

spread_data$bridge <- spread_data %>% with(ifelse(!is.na(fir)&is.na(sec)&!is.na(thi),1,
                                                  ifelse(!is.na(fir)&is.na(thi)&!is.na(fou),1,
                                                         ifelse(!is.na(sec)&is.na(thi)&!is.na(fou),1,0))))
spread_data1$bridge <- spread_data1 %>% with(ifelse(!is.na(fir)&is.na(sec)&!is.na(thi),1,
                                                    ifelse(!is.na(fir)&is.na(thi)&!is.na(fou),1,
                                                           ifelse(!is.na(sec)&is.na(thi)&!is.na(fou),1,0))))

frequency_bridge <- cbind(bssid = levels(total_data$bssid), bridge = 0, frequency_24 = 0, rssi = 0, pid = nowpid) %>% as.data.frame
frequency_bridge$frequency_24 <- ifelse(apply(spread_data[,1:4],1,mean,na.rm=T)<3000,1,0) %>% as.vector
frequency_bridge$rssi <- apply(spread_data1[,1:4],1,mean,na.rm=T) %>% as.vector
frequency_bridge$bridge <- spread_data$bridge
save_bridge <- save_bridge %>% bind_rows(frequency_bridge)

# # 중간에 사라졌다가 다시 생긴 와이파이 : bridge=1, 주파수 2.4GHz : frequency_24=1, rssi=rssi 평균
# ### 여기서 결과값은 4번째, 5번째, 6번쨰이다.
# 
# ## 주파수, bridge => 관계있음
# ### bridge==0일 경우에 주파수는 2.4Ghz가 더 많다. bridge==1일 경우에 주파수의 개수는 동일
# table(frequency_bridge$bridge,frequency_bridge$frequency_24)
# #    0  1
# # 0 19 69
# # 1 10 11
# 
# #    0  1
# # 0 74 96
# # 1  7 12
# 
# #    0  1
# # 0 47 63
# # 1  6 10
# cor.test(frequency_bridge$bridge,frequency_bridge$frequency_24)
# # -0.2323005
# # 0.0406348
# # 0.03525593
# 
# ## 주파수, rssi => 상관없음
# frequency_bridge[frequency_bridge$frequency_24==1,]$rssi %>% summary
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -93.00  -88.00  -83.29  -82.15  -78.75  -50.50 
# 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -91.00  -84.00  -79.00  -78.34  -74.56  -45.25 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -95.00  -81.50  -75.00  -74.54  -70.00  -36.25 
# 
# frequency_bridge[frequency_bridge$frequency_24==0,]$rssi %>% summary
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -89.00  -87.00  -84.00  -80.16  -70.00  -57.00
# 
# #    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# #  -91.00  -87.00  -85.00  -82.03  -82.33  -37.00 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -90.00  -87.00  -83.00  -80.14  -76.00  -60.50
# cor.test(frequency_bridge$frequency_24,frequency_bridge$rssi)
# # -0.107065
# # 0.2026082
# # 0.2545193
# 
# t.test(frequency_bridge[frequency_bridge$frequency_24==0,]$rssi,frequency_bridge[frequency_bridge$frequency_24==1,]$rssi)
# # 평균이 다르다.
# 
# ## bridge, rssi => 매우 상관있음 => 
# frequency_bridge[frequency_bridge$bridge==1,]$rssi %>% summary
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -88.50  -87.00  -85.00  -85.03  -83.67  -77.00
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -88.50  -84.50  -82.50  -81.82  -79.00  -73.00 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -88.00  -81.54  -79.58  -79.71  -77.88  -72.33 
# # => bridge 잘못 와이파이 기록된 것은 rssi가 낮을 경우로 한정된다.
# 
# frequency_bridge[frequency_bridge$bridge==0,]$rssi %>% summary
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -93.00  -87.25  -83.00  -80.81  -75.75  -50.50 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -91.00  -86.00  -82.33  -79.71  -76.33  -37.00 
# #   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# # -95.00  -86.00  -78.12  -76.49  -70.12  -36.25 
# cor.test(frequency_bridge$bridge,frequency_bridge$rssi)
# # -0.2027575
# # -0.07044411
# # -0.09862063
# 
# t.test(frequency_bridge[frequency_bridge$bridge==0,]$rssi,frequency_bridge[frequency_bridge$bridge==1,]$rssi)
# # 평균이 같다.




# date -> factor형 변환
total_data_change <- total_data
total_data_change$date <- as.factor(total_data_change$date)


### wifi bssid plot 1


geom.text.size = 1.8

png(filename=paste0(nowpid,"_tileplot.png"))

total_data_change %>% ggplot(aes(x=bssid,y=date,color="red")) +
  geom_tile(aes(fill = as.numeric(frequency))) +
  geom_text(aes(x=bssid, y=date,label=as.numeric(rssi)),  size=geom.text.size, color = "white") +
  theme(legend.position = "none" ,
        strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1))

dev.off()


### wifi bssid plot 2


bssid_unique2 <- total_data_change %>% subset(index==1) %>% .$bssid %>% unique
total_data_change2 <-  total_data_change %>% subset(bssid %in% bssid_unique2)

total_data_change2 %>% ggplot(aes(x=bssid,y=date,color="red")) +
  geom_tile(aes(fill = as.numeric(frequency))) +
  geom_text(aes(x=bssid, y=date,label=as.numeric(rssi)),  size=geom.text.size, color = "white") +
  theme(legend.position = "none" ,
        strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1))


### wifi bssid plot 3


bssid_index3 <- total_data_change %>% .$index %>% unique %>% .[2]
bssid_unique3 <- total_data_change %>% subset(index==bssid_index3) %>% .$bssid
total_data_change3 <-  total_data_change %>% subset(bssid %in% bssid_unique3)

total_data_change3 %>% ggplot(aes(x=bssid,y=date,color="red")) +
  geom_tile(aes(fill = as.numeric(frequency))) +
  geom_text(aes(x=bssid, y=date,label=as.numeric(rssi)),size=geom.text.size, color = "white") +
  theme(legend.position = "none" ,
        strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1))



### wifi bssid plot 4


bssid_index4 <- total_data_change %>% .$index %>% unique %>% .[3]
bssid_unique4 <- total_data_change %>% subset(index==bssid_index4) %>% .$bssid
total_data_change4 <-  total_data_change %>% subset(bssid %in% bssid_unique4)

total_data_change4 %>% ggplot(aes(x=bssid,y=date,color="red")) +
  geom_tile(aes(fill = as.numeric(frequency))) +
  geom_text(aes(x=bssid, y=date,label=as.numeric(rssi)),  size=geom.text.size, color = "white") +
  theme(legend.position = "none" ,
        strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1))




### wifi bssid plot 5

bssid_index4 <- total_data_change %>% .$index %>% unique %>% last
bssid_unique4 <- total_data_change %>% subset(index==bssid_index4) %>% .$bssid
total_data_change4 <-  total_data_change %>% subset(bssid %in% bssid_unique4)

total_data_change4 %>% ggplot(aes(x=bssid,y=date,color="red")) +
  geom_tile(aes(fill = as.numeric(frequency))) +
  geom_text(aes(x=bssid, y=date,label=as.numeric(rssi)),  size=geom.text.size, color = "white") +
  theme(legend.position = "none" ,
        strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1))





## 여기서 나는 무엇을 하고 싶은가?
#### 내부 수집에서 나타난 와이파이 10개 중에 6개만 검출되면 1/2 * 6/10, 
#### 전체 와이파이 10개에서 새로운 와이파이 4개가 잡히면 
#### 1/2 * (1-4/10) 식으로 가중치를 줘서 값을 구할 수 있지 않을까?

}

frequency_bridge <- save_bridge

# 중간에 사라졌다가 다시 생긴 와이파이 : bridge=1, 주파수 2.4GHz : frequency_24=1, rssi=rssi 평균
### 여기서 결과값은 4번째, 5번째, 6번쨰이다.

## 주파수, bridge => 관계없음
### bridge==0일 경우에 주파수는 2.4Ghz가 더 많다. bridge==1일 경우에 주파수의 개수는 동일
table(frequency_bridge$bridge,frequency_bridge$frequency_24)
cor.test(frequency_bridge$bridge,frequency_bridge$frequency_24)

## 주파수, rssi => 상관없음
frequency_bridge[frequency_bridge$frequency_24==1,]$rssi %>% summary


frequency_bridge[frequency_bridge$frequency_24==0,]$rssi %>% summary
cor.test(frequency_bridge$frequency_24,frequency_bridge$rssi)

t.test(frequency_bridge[frequency_bridge$frequency_24==0,]$rssi,frequency_bridge[frequency_bridge$frequency_24==1,]$rssi)

## bridge, rssi => 매우 상관있음 => 
frequency_bridge[frequency_bridge$bridge==1,]$rssi %>% summary
# => bridge 잘못 와이파이 기록된 것은 rssi가 낮을 경우로 한정된다.

frequency_bridge[frequency_bridge$bridge==0,]$rssi %>% summary
cor.test(frequency_bridge$bridge,frequency_bridge$rssi)

t.test(frequency_bridge[frequency_bridge$bridge==0,]$rssi,frequency_bridge[frequency_bridge$bridge==1,]$rssi)

# 주파수 고려안해줄거임
# # => bridge 잘못 와이파이 기록된 것은 rssi가 낮을 경우로 한정된다.
# # 이것을 명명하자.
# # 와이파이가 사라졌다 다시 생긴 것을 rssi가 낮은 수치에서 발생할 수 있다고 판단한다. 
# # 그러면 70이하인 애들만 드려보자...!! (-70이상)