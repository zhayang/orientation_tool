% distance and azimuth between two coordinates in degrees
% input can be vectors
function [dist, az]= coord2dist(lat1,lon1,lat2,lon2)
lat1rad  =  lat1*pi/180;
lat2rad  =  lat2*pi/180;
lon1rad  =  lon1*pi/180;
lon2rad  =  lon2*pi/180;

dlat = lat2rad-lat1rad;
dlon = lon2rad-lon1rad;

a=sin(dlat/2).^2+cos(lat1rad).*cos(lat2rad).*sin(dlon/2).^2;
c=2*atan2(sqrt(a),sqrt(1-a));
dist=c*180/pi;

y = sin(dlon).*cos(lat2rad);
x = cos(lat1rad).*sin(lat2rad) - sin(lat1rad).*cos(lat2rad).*cos(dlon);
az = atan2(y,x)*180/pi;

az=mod(az+360,360);

return    
end