CREATE TABLE adtrack.train (
    ip INT(11) NOT NULL,
    app INT(5) NOT NULL,
    device INT(5) NOT NULL,
    os INT(5) NOT NULL,
    channels INT(5) NOT NULL,
    click_time datetime NOT NULL,
    attributed_time	datetime,
	is_attributed int(2) NOT NULL
);

CREATE TABLE adtrack.test (
	click_id INT(11) NOT NULL,
    ip INT(11) NOT NULL,
    app INT(5) NOT NULL,
    device INT(5) NOT NULL,
    os INT(5) NOT NULL,
    channels INT(5) NOT NULL,
    click_time datetime NOT NULL
);


CREATE TABLE adtrack.test_supplement (
	click_id INT(11) NOT NULL,
    ip INT(11) NOT NULL,
    app INT(5) NOT NULL,
    device INT(5) NOT NULL,
    os INT(5) NOT NULL,
    channels INT(5) NOT NULL,
    click_time datetime NOT NULL
);

CREATE TABLE adtrack.total (
    ip INT(11) NOT NULL,
    app INT(5) NOT NULL,
    device INT(5) NOT NULL,
    os INT(5) NOT NULL,
    channels INT(5) NOT NULL,
    click_time datetime NOT NULL
);


LOAD DATA LOCAL INFILE "/home/rjs/바탕화면/adtrack/data/train.csv" 
INTO TABLE adtrack.train
fields terminated by ','
lines terminated by '\n'
IGNORE 1 ROWS
(ip,app,device,os,channels,click_time,attributed_time,is_attributed);


LOAD DATA LOCAL INFILE "/home/rjs/바탕화면/adtrack/data/test.csv" 
INTO TABLE adtrack.test
fields terminated by ','
lines terminated by '\n'
IGNORE 1 ROWS
(click_id,ip,app,device,os,channels,click_time);

LOAD DATA LOCAL INFILE "/home/rjs/바탕화면/adtrack/data/test_supplement.csv" 
INTO TABLE adtrack.test_supplement
fields terminated by ','
lines terminated by '\n' 
IGNORE 1 ROWS
(click_id,ip,app,device,channels,os,click_time);

CREATE TABLE adtrack.new_train LIKE adtrack.train;
CREATE TABLE adtrack.new_test LIKE adtrack.test;
CREATE TABLE adtrack.new_test_supplement LIKE adtrack.test_supplement;

INSERT INTO adtrack.total(ip, app, device, os, channels, click_time)
SELECT ip,app,device,os,channels,click_time 
FROM adtrack.train 
UNION ALL 
SELECT ip,app,device,os,channels,click_time 
FROM adtrack.test_supplement;

select ip,app,count(channels) as ip_app_count from adtrack.test group by ip,app

(select ip,app,count(channels) as ip_app_count from adtrack.test group by ip,app) as ip_app_count;
select ip,app,count(channels) as ip_app_count from adtrack.test group by ip,app;
select ip,app,count(channels) as ip_app_count from adtrack.test group by ip,app;
select ip,app,count(channels) as ip_app_count from adtrack.test group by ip,app;

SELECT * FROM Table_Name ORDER BY RAND() LIMIT 0,10;

select name, phone, selling
from adtrack.total join demo_property
on demo_people.pid = demo_property.pid;

select 
    *
from
    (select
        distinct adtrack.test.click_id as test_click_id,
        adtrack.test_supplement.click_id as sup_click_id
    from adtrack.test_supplement inner join adtrack.test on 
adtrack.test_supplement.click_time = adtrack.test.click_time and
adtrack.test_supplement.channels = adtrack.test.channels and
adtrack.test_supplement.app = adtrack.test.app  and
adtrack.test_supplement.ip = adtrack.test.ip and
adtrack.test_supplement.device = adtrack.test.device and
adtrack.test_supplement.os = adtrack.test.os) x
where test_click_id is not null order by test_click_id;