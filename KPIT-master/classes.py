class car:
	def __init__(self,coordinate,velocity,number=0):
		self.x=coordinate[0]
		self.y=coordinate[1]
		self.cluster=number
		self.velocity =velocity
	def __str__(self):
		string="coord "+ str(self.x)+" "+str(self.y)
		return string

# a=car([0,0],56)

# print a

class cluster:

	def __init__(self,cars,centroid,number,weight=0):
		self.cars=cars
		self.weight=weight
		self.avg_vel=0
		self.centroid =centroid
		self.cluster_no=number

	def assign_avg_vel(self) :
		self.avg_vel=sum([x.velocity for x in self.cars])
		self.avg_vel/=float(len(self.cars))
	

	def assign_weight(self,d) :
		self.weight=len(self.cars)/(d/self.avg_vel)
	def __str__(self):
		# for car in self.cars:
		# 	print car,
		string="wight:"+str(self.weight)+"number: "+str(self.cluster_no)+ "avg_vel:"+str(self.avg_vel)
		return string

# b=cluster(a,[0,0],0)
# print b

def calc_weight_of_lane(clusters):
	total_weight=sum([x.weight for x in clusters])
	avg_weight=total_weight/len(cars)
	return total_weight, avg_weight
'''
def calc_total_weight(lanes):
	return 
'''

def distribute_time(threshold,T0,weight_list):
	sum_weight=sum(weight_list)

	return [(weight_list[m]/sum_weight)*T0 for m in range(0,4)]

		

