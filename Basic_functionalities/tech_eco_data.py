from scipy.interpolate import interp1d
import numpy as np

def get_biogas_share_in_network_RTE(year): 
	return np.interp(year, [2019, 2030, 2040, 2050], [0] * 4)#[.001, .11, .37, 1])

def get_capex_new_tech_RTE(tech, hyp='ref', year=2020, var=None): 
	# https://assets.rte-france.com/prod/public/2022-06/FE2050%20_Rapport%20complet_ANNEXES.pdf page 937 
	years = [2020, 2030, 2040, 2050, 2060]


	if tech == "CCG":
			capex = {
				'ref':  interp1d(years, [900]*5, fill_value=(0,900),bounds_error=False),
				'low':  interp1d(years, [900]*5, fill_value=(0,900),bounds_error=False),
				'high': interp1d(years, [900]*5, fill_value=(0,900),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [40]*5, fill_value=(101,40),bounds_error=False),
				'low': interp1d(years, [40]*5,fill_value=(101,40),bounds_error=False),
				'ref': interp1d(years, [40]*5,fill_value=(101,40),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [40]*5,fill_value=(20,40),bounds_error=False),
				'low':  interp1d(years, [40]*5,fill_value=(20,40),bounds_error=False),
				'ref':  interp1d(years, [40]*5,fill_value=(20,40),bounds_error=False),
			}

	elif tech == "TAC":
			capex = {
				'ref':  interp1d(years, [600]*5,fill_value=(0,600),bounds_error=False),
				'low':  interp1d(years, [600]*5,fill_value=(0,600),bounds_error=False),
				'high': interp1d(years, [600]*5,fill_value=(0,600),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [20]*5,fill_value=(74,20),bounds_error=False),
				'low': interp1d(years,  [20]*5,fill_value=(74,20),bounds_error=False),
				'ref': interp1d(years, [20]*5,fill_value=(74,20),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
				'low':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
				'ref':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
			}

	elif tech == "Coal_p":
			capex = {
				'ref':  interp1d(years, [1100]*5,fill_value=(0,1100),bounds_error=False),
				'low':  interp1d(years, [1100]*5,fill_value=(0,1100),bounds_error=False),
				'high': interp1d(years, [1100]*5,fill_value=(0,1100),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [40]*5,fill_value=(124,40),bounds_error=False),
				'low': interp1d(years, [40]*5,fill_value=(124,40),bounds_error=False),
				'ref': interp1d(years, [40]*5,fill_value=(124,40),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
				'low':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
				'ref':  interp1d(years, [30]*5,fill_value=(15,30),bounds_error=False),
			}

	elif tech == "OldNuke":
			capex = {
				'ref':  interp1d(years, [0]*5,fill_value=(0,0),bounds_error=False),
				'low':  interp1d(years, [0]*5,fill_value=(0,0),bounds_error=False),
				'high': interp1d(years, [0]*5,fill_value=(0,0),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [186]*5,fill_value=(186,186),bounds_error=False),
				'low': interp1d(years, [186]*5,fill_value=(186,186),bounds_error=False),
				'ref': interp1d(years, [186]*5,fill_value=(186,186),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
				'low':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
				'ref':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
			}

	elif tech == "HydroReservoir":
			capex = {
				'ref':  interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
				'low':  interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
				'high': interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [15]*5,fill_value=(121,15),bounds_error=False),
				'low': interp1d(years, [15]*5,fill_value=(121,15),bounds_error=False),
				'ref': interp1d(years, [15]*5,fill_value=(121,15),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [70]*5,fill_value=(40,70),bounds_error=False),
				'low':  interp1d(years, [70]*5,fill_value=(40,70),bounds_error=False),
				'ref':  interp1d(years, [70]*5,fill_value=(40,70),bounds_error=False),
			}

	elif tech == "HydroRiver":
			capex = {
				'ref':  interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
				'low':  interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
				'high': interp1d(years, [1000]*5,fill_value=(0,1000),bounds_error=False),
			}
			opex = {
				'high': interp1d(years,[15]*5,fill_value=(121,15),bounds_error=False),
				'low': interp1d(years, [15]*5,fill_value=(121,15),bounds_error=False),
				'ref': interp1d(years, [15]*5,fill_value=(121,15),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [70]*5,fill_value=(40,70),bounds_error=False),
				'low':  interp1d(years,[70]*5,fill_value=(40,70),bounds_error=False),
				'ref':  interp1d(years,[70]*5,fill_value=(40,70),bounds_error=False),
			}


	elif tech == "NewNuke":
			capex = {
				'ref':  interp1d(years, [11900, 11900, 5500, 5000, 5000],fill_value=(11900,5000),bounds_error=False),
				'low':  interp1d(years, [11900, 11900, 5035, 4500, 4500],fill_value=(11900,4500),bounds_error=False),
				'high': interp1d(years, [11900, 11900, 7900, 7900, 7900],fill_value=(11900,7900),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [100]*5,fill_value=(100,100),bounds_error=False),
				'low': interp1d(years, [100]*5,fill_value=(100,100),bounds_error=False),
				'ref': interp1d(years, [100]*5,fill_value=(100,100),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
				'low':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
				'ref':  interp1d(years, [60]*5,fill_value=(60,60),bounds_error=False),
			}

	elif tech == "WindOffShore":
			capex = {
				'ref':  interp1d(years, [2600, 1700, 1500, 1300, 1300],fill_value=(2600,1300),bounds_error=False),
				'low':  interp1d(years, [2600, 1300, 1000, 700, 700],fill_value=(2600,700),bounds_error=False),
				'high': interp1d(years, [2600, 2100, 2000, 1900, 1900],fill_value=(2600,1900),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [80, 65, 60, 55, 55],fill_value=(80,55),bounds_error=False),
				'low': interp1d(years,  [80, 54, 38, 28, 28],fill_value=(80,28),bounds_error=False),
				'ref': interp1d(years,  [80, 58, 47, 36, 36],fill_value=(80,26),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
				'low':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
				'ref':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
			}

	elif tech == "WindOffShore_flot":
			capex = {
				'ref':  interp1d(years, [3100, 2500, 2200, 1900, 1900],fill_value=(3100,1900),bounds_error=False),
				'low':  interp1d(years, [3100, 2100, 1700, 1300, 1300],fill_value=(3100,1300),bounds_error=False),
				'high': interp1d(years, [3100, 2900, 2700, 2500, 2500],fill_value=(3100,2500),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [110, 90, 80, 70, 70],fill_value=(110,70),bounds_error=False),
				'low': interp1d(years,  [110, 75, 50, 40, 40],fill_value=(110,40),bounds_error=False),
				'ref': interp1d(years,  [110, 80, 60, 50, 50],fill_value=(110,50),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
				'low':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
				'ref':  interp1d(years, [20, 25, 30, 40, 40],fill_value=(20,40),bounds_error=False),
			}

	elif tech == "WindOnShore":
			capex = {
				'ref':  interp1d(years, [1300, 1200, 1050, 900, 900],fill_value=(0,900),bounds_error=False),
				'low':  interp1d(years, [1300, 710, 620, 530, 530],fill_value=(0,530),bounds_error=False),
				'high': interp1d(years, [1300, 1300, 1300, 1300, 1300],fill_value=(0,1300),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [40, 40, 40, 40, 40],fill_value=(168,40),bounds_error=False),
				'low': interp1d(years,  [40, 22, 18, 16, 16],fill_value=(168,16),bounds_error=False),
				'ref': interp1d(years,  [40, 35, 30, 25, 25],fill_value=(168,25),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(20,30),bounds_error=False),
				'low':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(20,30),bounds_error=False),
				'ref':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(20,30),bounds_error=False),
			}

	elif tech == "Solar":
			capex = {
				'ref':  interp1d(years, [747, 597, 517, 477, 477],fill_value=(0,477),bounds_error=False),
				'low':  interp1d(years, [747, 557, 497, 427, 427],fill_value=(0,127),bounds_error=False),
				'high': interp1d(years, [747, 612, 562, 527, 527],fill_value=(0,527),bounds_error=False),
			}
			opex = {
				'high': interp1d(years, [11, 10, 10, 9, 9],fill_value=(227,9),bounds_error=False),
				'low': interp1d(years,  [11, 9, 8, 7, 7],fill_value=(227,7),bounds_error=False),
				'ref': interp1d(years,  [11, 10, 9, 8, 8],fill_value=(227,8),bounds_error=False),
			}
			life = {
				'high':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(15,30),bounds_error=False),
				'low':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(15,30),bounds_error=False),
				'ref':  interp1d(years, [25, 30, 30, 30, 30],fill_value=(15,30),bounds_error=False),
			}

	elif tech == "Electrolysis":
			capex = {
				'ref':  interp1d(years, [1313, 641, 574, 507, 440],fill_value=(1313,440),bounds_error=False),
			}
			opex = {
				'ref': interp1d(years, [12] *5,fill_value=(12,12),bounds_error=False),
			}
			life = {
				'ref':  interp1d(years, [20] * 5,fill_value=(20,20),bounds_error=False),
			}

	elif tech == 'Battery - 1h': 
			capex = {
				'ref':  interp1d(years, [537, 406, 332, 315, 315],fill_value=(537,315),bounds_error=False), # EUR/kW
			}		

			opex = {
				'ref':  interp1d(years, [11] * 5,fill_value=(11,11),bounds_error=False), # EUR/kW/yr
			}	
			life = {
				'ref':  interp1d(years, [15] * 5,fill_value=(15,15),bounds_error=False),
			}

	elif tech == 'Battery - 4h': 
			capex = {
				'ref':  interp1d(years, [1480, 1101, 855, 740, 740],fill_value=(1480,740),bounds_error=False), # EUR/kW
			}		

			opex = {
				'ref':  interp1d(years, [30] * 5,fill_value=(30,30),bounds_error=False), # EUR/kW/yr
			}	
			life = {
				'ref':  interp1d(years, [15] * 5,fill_value=(15,15),bounds_error=False),
			}

	elif tech == 'Salt cavern': 
			capex = {
				'ref':  interp1d(years, [350] * 5,fill_value=(350,350),bounds_error=False), # EUR/kWhLHV
			}		

			opex = {
				'ref':  interp1d(years, [2] * 5,fill_value=(2,2),bounds_error=False), # EUR/kW/yr
			}	
			life = {
				'ref':  interp1d(years, [40] * 5,fill_value=(40,40),bounds_error=False),
			}

	elif tech == 'tankH2_G':
			capex = {
				'ref':  interp1d(years, [18] * 5,fill_value=(18,18),bounds_error=False), # EUR/kWhLHV
			}

			opex = {
				'ref':  interp1d(years, [2] * 5,fill_value=(1,1),bounds_error=False), # EUR/kW/yr
			}
			life = {
				'ref':  interp1d(years, [20] * 5,fill_value=(20,20),bounds_error=False),
			}

	elif tech == 'STEP':
			capex = {
				'ref':  interp1d(years, [1000] * 5,fill_value=(0,1000),bounds_error=False), # EUR/kWhLHV
			}

			opex = {
				'ref':  interp1d(years, [15] * 5,fill_value=(15,15),bounds_error=False), # EUR/kW/yr
			}
			life = {
				'ref':  interp1d(years, [70] * 5,fill_value=(50,70),bounds_error=False),
			}

	elif tech == 'Interco':
			capex = {
				'ref':  interp1d(years, [0] * 5,fill_value=(0,0),bounds_error=False), # EUR/kWhLHV
			}

			opex = {
				'ref':  interp1d(years, [0] * 5,fill_value=(0,0),bounds_error=False), # EUR/kW/yr
			}
			life = {
				'ref':  interp1d(years, [100] * 5,fill_value=(100,100),bounds_error=False),
			}

	elif tech == 'curtailment':
			capex = {
				'ref':  interp1d(years, [0] * 5,fill_value=(0,0),bounds_error=False), # EUR/kWhLHV
			}

			opex = {
				'ref':  interp1d(years, [0] * 5,fill_value=(0,0),bounds_error=False), # EUR/kW/yr
			}
			life = {
				'ref':  interp1d(years, [100] * 5,fill_value=(100,100),bounds_error=False),
			}


	if var == "capex": 
		return 1e3 * capex[hyp](year)
	elif var == "opex": 
		return 1e3 * opex[hyp](year)
	elif var == 'lifetime': 
		return life[hyp](year)
	else: 
		return 1e3 * capex[hyp](year), 1e3 * opex[hyp](year), float(life[hyp](year))
