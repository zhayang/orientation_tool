% azimuth from two coordinates in degrees
function az= coord2az(lat1,lon1,lat2,lon2)
lat1rad  =  lat1*pi/180;
lat2rad  =  lat2*pi/180;
lon1rad  =  lon1*pi/180;
lon2rad  =  lon2*pi/180;

dlon = lon2rad-lon1rad;

y = sin(dlon)*cos(lat2rad);
x = cos(lat1rad)*sin(lat2rad) - sin(lat1rad)*cos(lat2rad)*cos(dlon);
az = atan2(y,x)*180/pi;
az=mod(az+360,360);
end