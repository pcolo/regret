# writen on python 3.4.2 win32
# Version 0. 2014-11-07.
# IAS

import math
import numpy
import pandas

def distance_on_unit_sphere(lat1, long1, lat2, long2):
	# source: http://www.johndcook.com/python_longitude_latitude.html

	# Convert latitude and longitude to spherical coordinates in radians.
	degrees_to_radians = math.pi/180.0
		
	# phi = 90 - latitude
	phi1 = (90.0 - lat1)*degrees_to_radians
	phi2 = (90.0 - lat2)*degrees_to_radians
		
	# theta = longitude
	theta1 = long1*degrees_to_radians
	theta2 = long2*degrees_to_radians
		
	# Compute spherical distance from spherical coordinates.
	cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
		math.cos(phi1)*math.cos(phi2))
	arc = math.acos( cos )
	return arc

# stations_within_r: returns the IDs of all stations within a distance r (km) of the center of
# the region defined by the $regionID
def stations_within_r(regionID, r):
	
	# read data from files
	regions = pandas.read_csv("./_RegionsData.csv")
	stations = pandas.read_csv("./_StationsData.csv")
	
	# extract latitude and longitude of the center of the region
	Rlatitude = float(regions[regions['ABR'] == regionID]['LATITUDE'])
	Rlongitude = float(regions[regions['ABR'] == regionID]['LONGITUDE'])
	
	# compute the distance to every station
	stations['DISTANCE'] = 0
	for ix in range(0, len(stations.index)-1):
		stations.loc[ix, 'DISTANCE'] = 6373*distance_on_unit_sphere(Rlatitude, Rlongitude,
			float(stations.loc[ix, 'LATITUDE']), float(stations.loc[ix, 'LONGITUDE']))
	
	# return stations IDs
	return stations.loc[stations['DISTANCE'] <= r, '\ufeffID'].tolist()

# stations_measurements: extract the measuremts of the stations in the $stations array
# from the $filename file
def stations_measurements(stations, filename):
	
	# read data from file
	allstationsts = pandas.read_csv("./tsfiles/" + filename)
	
	# find out which stations have this kind of measurement
	stations = list( set(allstationsts.columns.tolist()) & set(stations) )
	
	# extract stations columns to a smaller array 
	stationsts = allstationsts.loc[:,['DAYNUM','PERIOD']+stations].values
	del allstationsts
	
	# reshape measurements array
	newstationsts = numpy.zeros([numpy.max(stationsts[:,0]),
		(len(stationsts[0])-2)*numpy.max(stationsts[:,1])])
	DJ = numpy.max(stationsts[:,1])
	stationsts[:,0] = numpy.max(stationsts[:,0]) - stationsts[:,0]
	stationsts[:,1] = stationsts[:,1] - 1
	for jx in range(2, len(stationsts[0])):
		J0 = (jx-2)*DJ
		for ix in range(len(stationsts)):
			newstationsts[int(stationsts[ix,0]), int(J0+stationsts[ix, 1])] = stationsts[ix, jx]
	
	# write column names
	colnames = []
	per = 0
	for s in stations:
		for per in range(0, int(numpy.max(stationsts[:,1])+1)):
			colnames.append(s + '.' + str(per + 1).zfill(2))
	
	# return object with relevant info
	return { 'data': newstationsts, 'colnames': colnames, 'fstations': stations, 'periods': per+1 }

#####################################################################################################

# solarpower_D: returns the 'day D' array (wrapper of stations_measurements)
def solarpower_D(regionID):
	newregionts = stations_measurements([regionID], "SolarPowRatio_BE-DE-FR_30min_2013.csv")['data']
	return(newregionts[range(len(newregionts)-1),:])
# test: numpy.savetxt("BY.csv", solarpower_D("BY"), delimiter=",")

# solarpower_meteo_D_1: returns the 'day D-1' array
def solarpower_meteo_D_1(regionID, r = 100):
	
	# extract solar production data
	aux = stations_measurements([regionID], "SolarPowRatio_BE-DE-FR_30min_2013.csv")
	exportarray = aux['data']
	colnames = numpy.core.defchararray.add(["PV."]*aux['periods'], aux['colnames']).tolist()
	del aux
	
	# extract dates
	aux = pandas.read_csv("./tsfiles/_DatesBackward.csv")
	exportarray = numpy.concatenate((exportarray, aux.values), axis=1)
	colnames = colnames + ['YEAR','MONTH','DAY']
	del aux
	
	# extract clear sky radiation data
	aux = stations_measurements([regionID], "ClearSkyRadiation_BE-DE-FR_30min_2013.csv")
	exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
	colnames = colnames + numpy.core.defchararray.add(["CSR."]*aux['periods'],
		aux['colnames']).tolist()
	del aux
	
	# extract solar azimuth and solar zenith data
	aux = stations_measurements([regionID], "SolarAzimuth_BE-DE-FR_30min_2013.csv")
	exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
	colnames = colnames + numpy.core.defchararray.add(["Azimuth."]*aux['periods'],
		aux['colnames']).tolist()
	del aux
	aux = stations_measurements([regionID], "SolarZenith_BE-DE-FR_30min_2013.csv")
	exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
	colnames = colnames + numpy.core.defchararray.add(["Zenith."]*aux['periods'],
		aux['colnames']).tolist()
	del aux
	
	# determine the meteorological stations within a radius of $r km
	stations = stations_within_r(regionID, r)
	
	# extract wind speed and wind direction data (meteo stations)
	auxS = stations_measurements(stations, "WindMeanSpeed_BE-DE-FR_hourly_2013.csv")
	auxD = stations_measurements(stations, "WindDirection_BE-DE-FR_hourly_2013.csv")
	cstations = list( set(auxS['fstations']) & set(auxD['fstations']) )
	if len(cstations) > 0:
		cstations.sort()
		for s in cstations:
			wsj = auxS['fstations'].index(s)
			wdj = auxD['fstations'].index(s)
			exportarray = numpy.concatenate((exportarray, 
				auxS['data'][:,(wsj*auxS['periods']):((wsj+1)*auxS['periods'])]), axis=1)
			colnames = colnames + numpy.core.defchararray.add(["WindSpeed."]*auxS['periods'],
				auxS['colnames'][(wsj*auxS['periods']):((wsj+1)*auxS['periods'])]).tolist()
			exportarray = numpy.concatenate((exportarray, 
				auxD['data'][:,(wdj*auxD['periods']):((wdj+1)*auxD['periods'])]), axis=1)
			colnames = colnames + numpy.core.defchararray.add(["WindDir."]*auxD['periods'],
				auxD['colnames'][(wdj*auxD['periods']):((wdj+1)*auxD['periods'])]).tolist()
		del cstations, s, wsj, wdj
	del auxS, auxD
	
	# extract cloud coverage data (meteo stations)
	aux = stations_measurements(stations, "CloudCoverage_BE-DE-FR_hourly_2013.csv")
	if len(aux['data'][0]) > 0:
		exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
		colnames = colnames + numpy.core.defchararray.add(["CloudCover."]*len(aux['colnames']),
			aux['colnames']).tolist()
	del aux
	
	# extract temperature (meteo stations)
	aux = stations_measurements(stations, "Temperature_BE-DE-FR_hourly_2013.csv")
	if len(aux['data'][0]) > 0:
		exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
		colnames = colnames + numpy.core.defchararray.add(["Temp."]*len(aux['colnames']),
			aux['colnames']).tolist()
	del aux
	
	# extract pressure (meteo stations)
	aux = stations_measurements(stations, "Pressure_BE-DE-FR_hourly_2013.csv")
	if len(aux['data'][0]) > 0:
		exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
		colnames = colnames + numpy.core.defchararray.add(["Pressure."]*len(aux['colnames']),
			aux['colnames']).tolist()
	del aux
	
	# extract humidity (meteo stations)
	aux = stations_measurements(stations, "Humidity_BE-DE-FR_hourly_2013.csv")
	if len(aux['data'][0]) > 0:
		exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
		colnames = colnames + numpy.core.defchararray.add(["Humidity."]*len(aux['colnames']),
			aux['colnames']).tolist()
	del aux
	
	# extract precipitation (meteo stations)
	aux = stations_measurements(stations, "Precipitation_BE-DE-FR_hourly_2013.csv")
	if len(aux['data'][0]) > 0:
		exportarray = numpy.concatenate((exportarray, aux['data']), axis=1)
		colnames = colnames + numpy.core.defchararray.add(["PCPN."]*len(aux['colnames']),
			aux['colnames']).tolist()
	del aux
	
	return {'data': exportarray[1:len(exportarray),:], 'colnames': colnames, 'stations': stations}
#test: numpy.savetxt("BY.csv", solarpower_meteo_D_1("BY")['data'], delimiter=",")
#to obtain numerical array: solarpower_meteo_D_1($regionID, $radious)['data']

# exec(open("./loadRegion.py").read())