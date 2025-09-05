with t1 as(
select SearchCityName, any(SearchGuid) as SearchGuid 
from searches.hotel_searches_booked hsb
where  SearchDate >= '2025-07-15' and
 UPPER(SearchCityName) in 
('BROOKLYN', 'NAPERVILLE', 'REDMOND', 'CHESTERFIELD', 'ONTARIO', 
'SEATTLE', 'NEW YORK CITY', 'LOS ANGELES', 'DUBAI', 'BANGKOK', 'LAS VEGAS')
GROUP by SearchCityName),
t2 as (select SearchGuid from t1),
l1 as(SELECT SearchGuid, min(FinalPrice) as CheapestPrice_Search,
avg(case when positionCaseInsensitive(HotelCompleteAddress, SearchCityName)>0 then HotelLatitude end) as lt1, 
avg(case when positionCaseInsensitive(HotelCompleteAddress, SearchCityName)>0 then HotelLongitude end) as ln1,
avg(case when lower(SearchCityName) = lower(HotelCityName) then HotelLatitude end) as lt2, 
avg(case when lower(SearchCityName) = lower(HotelCityName) then HotelLongitude end) as ln2, 
avg(HotelLatitude) as lt3, avg(HotelLongitude) as ln3
from searches.hotel_searches_booked hsb 
where SearchGuid in (select SearchGuid from t2) group by SearchGuid),
mains as(
    select `SearchDate`, `PortalID`, `CheckIn`, `CheckOut`,date_diff('days', SearchDate, CheckIn) as LeadDays,date_diff('days', CheckIn, CheckOut) as NumberOfNights
    , upper(`SearchCityName`) as SearchCityName, SearchLocationID, `SearchFrom`, `TotalRooms`, TotalChildren + TotalAdults as TotalPax, hsb.SearchGuid, `TripType`
    , `FPHotelID`, `IsBooked`, upper(`BrandName`) as BrandName,`HotelName`, upper(`HotelCityName`) as HotelCityName, upper(`HotelStateName`) as HotelStateName, upper(`HotelCountryCode`) as HotelCountryCode ,if(HotelCountryCode='US',1,0) as IsUS,`HotelCompleteAddress`
    , `HotelLatitude`, `HotelLongitude`,`HotelPropertyType`, `BookingEngine`, `TotalTax`, `SupplierPrice`, `DisplayPrice`, `FinalPrice`, FinalPrice / TotalRooms as FinalPrice_Per_Room
    ,`TotalServiceFee`, `ServiceFee`,`IsSplashHotel`, `FareportalMarkupEngine`, `SearchLatitude`, `SearchLongitude`, DisplayPrice - SupplierPrice as PriceMargin
    ,has(AmenitiesKnownName,'Bar') as IsBar, `SearchAirportCityCode`,`SearchLocationID`,`Past15_Bookings`
    , COALESCE(lt1,lt2,lt3) as mean_latitude, COALESCE(ln1,ln2,ln3) mean_longitude, FinalPrice - CheapestPrice_Search as CheapestHotel_PriceDiff, CheapestPrice_Search
    , greatCircleDistance(HotelLatitude,HotelLongitude,mean_latitude,mean_longitude) dcc, positionCaseInsensitive(HotelCityName, SearchCityName) + positionCaseInsensitive(SearchCityName,HotelCityName) > 0 as SameHotelCity
    , DENSE_RANK() Over (PARTITION by SearchGuid ORDER by FinalPrice) as FinalPrice_rank
    , DENSE_RANK() Over (PARTITION by SearchGuid ORDER by TotalTax) as TotalTax_rank
    , DENSE_RANK() Over (PARTITION by SearchGuid ORDER by dcc) as Distance_from_City_Center_rank
    , ROW_NUMBER() OVER (PARTITION by SearchGuid ORDER by rand() ) as rn
    , RecommenderPredictedScore, RecommenderPredictedRank, CMSHotelRank, PackageType
    from searches.hotel_searches_booked hsb
    left join l1 on l1.SearchGuid = hsb.SearchGuid 
    where hsb.SearchGuid in (select SearchGuid from t2))    
 SELECT SearchDate, PortalID, CheckIn, CheckOut, LeadDays, NumberOfNights, SearchCityName, SameHotelCity,SearchLocationID, SearchFrom, TotalRooms, TotalPax, SearchGuid, 
 RecommenderPredictedScore, RecommenderPredictedRank, TripType, FPHotelID, CMSHotelRank, PackageType, 
 IsBooked, BrandName, HotelName, HotelCityName, HotelStateName, HotelCountryCode, IsUS, HotelCompleteAddress, 
 HotelLatitude, HotelLongitude, HotelPropertyType, BookingEngine, TotalTax, SupplierPrice, DisplayPrice, 
 FinalPrice, FinalPrice_Per_Room, TotalServiceFee, ServiceFee, IsSplashHotel, FareportalMarkupEngine, 
 SearchLatitude, SearchLongitude, PriceMargin, IsBar, SearchAirportCityCode, 
 Past15_Bookings, mean_latitude, mean_longitude, CheapestHotel_PriceDiff, 
 CheapestPrice_Search, FinalPrice_rank, TotalTax_rank, Distance_from_City_Center_rank, 
 dcc/1000 as Distance_from_City_Center from mains;
