import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys,os
import random
import torchvision.transforms as tvt
sys.path.append(os.path.realpath('..'))

def save_img(name,img):
	plt.imsave(name,img)
	#print("Saved as {0}".format(name))

def disp_img(img):
	plt.imshow(img,cmap='Greys',interpolation='none')
	plt.show()

def make_img(out):
	img=np.max(out,axis=2)
	#plt.imshow(img,cmap='Greys')
	#plt.show()
	return img

def read_img(n,base,name):
	return plt.imread(os.path.join(base,'lots_of','pyr_img',name))

def get_patch(og,img,lb,ub):
	y=np.argwhere(np.logical_and(img>lb,img<ub))
	pos=y[np.random.randint(y.shape[0])]
	print(pos)
	ref=plt.imread(og)
	res=ref[pos[0]-5:pos[0]+5,pos[1]-5:pos[1]+5,:]
	#plt.imshow(res)
	#plt.show()
	return (pos[0],pos[1],res)
def load_mat(n,base,name):
	return load_pyr(os.path.join(base,'lots_of','pyr_lep_data',name))

def load_pyr(fil):
	f= h5py.File(fil, 'r')
	print('Loaded')
	ind=np.array(list(f['pyr_lep']['indices']))
	val=np.array(list(f['pyr_lep']['values']))
	#plt.hist(val)
	#plt.show()
	print('Read')
	out=np.zeros(list(f['pyr_lep']['shape'])).reshape(-1)

	out[ind]=val

	out=out.reshape(list(f['pyr_lep']['shape']))
	out=out.transpose(2,1,0)
	return out

def stat_lep(i,out,lep_stat):
	for j in range(1,25,1):
		lep_stat[i,j]=(out[:,:,j-1]>0).sum()
	return

def get_dev(x):
    t=np.copy(x)
    t-=np.mean(t[8:12,8:12])
    return float(np.sum(np.abs(t[8:12,8:12])))

def proc_gray(x):
    t=np.copy(x)
    t-=np.mean(t[8:12,8:12])
    t/=np.sum(np.abs(t[8:12,8:12]))
    t+=0.5
    return x

def rgb2gray(x):
    return 0.299*x[:,:,0]+ 0.587*x[:,:,1] + 0.114*x[:,:,2]

def proc(x):
    t=np.copy(x)
    t=rgb2gray(t)
    return proc_gray(t)


def comp(t):
	nums=[0,0,0,0,0,0]
	#top deviation, bottom deviation, top top-bottom diff, bottom top-bottom diff, top left-right diff, bottom left-right diff
	nums[0]=np.mean(abs(t[8:10,8:12]-np.mean(t[8:10,8:12])))
	nums[1]=np.mean(abs(t[10:12,8:12]-np.mean(t[10:12,8:12])))
	nums[2]=np.mean(t[8,8:12])-np.mean(t[9,8:12])
	nums[3]=np.mean(t[10,8:12])-np.mean(t[11,8:12])
	nums[4]=np.mean(t[8:10,8:10])-np.mean(t[8:10,10:12])
	nums[5]=np.mean(t[10:12,8:10])-np.mean(t[10:12,10:12])
	return np.asarray(nums)


base=os.path.dirname(os.path.abspath(__file__))


'''
lep_stat=np.zeros((81,25))

for i in range(1,82,1):
	out=load_mat(i,base)
	print(" Parsed {0}".format(i),end='')
	stat_lep(i-1,out,lep_stat)
	lep_stat[i-1:,0]=(make_img(out)[:,:]==0).sum()
	print(' Done')
'''


'''

q=os.listdir('lots_of/pyr_img')
w=os.listdir('lots_of/pyr_lep_data')
top_bin=2000
dist=np.zeros((top_bin,101))
indx=np.load('bins.npy')[-(top_bin):]
print(indx.shape)
indlist=list(indx)
lost_edges=0
edges=0
subt_edges=0

subt_non=0
lost_non=0
non=0
print(len(q))
np.save('dist_save.npy',dist)
print("Saved")
cnt=0
lost_edges=0
subt_edges=0
for i in range(497):
	dist=np.load('dist_save.npy')
	print("Loaded array")
	img_name=q[i]
	mat_name=q[i].split('.')[0]+'.mat'
	mat_name=mat_name.split('img')[0]+'lep'+mat_name.split('img')[1]
	print("IMAGE "+str(i),img_name,mat_name)
	out=load_mat(i,base,mat_name)
	ref=read_img(i,base,img_name)
	refbw=rgb2gray(ref)
	print(ref.shape)
	for j in range(1,2,6):
		#num_to_sample=int(round(num*lep_stat[i-1,j]/lep_stat[:,j].sum(axis=0)))
		inds=np.argwhere(out[15:-15,15:-15,j-1]>0)

		#rand=np.asarray(random.sample(range(np.size(inds,0)),num_to_sample))
		
		#print(j,m,arr)
		for pos in inds:
			cnt+=1
			res=refbw[pos[0]+6:pos[0]+26,pos[1]+6:pos[1]+26]
			resc=ref[pos[0]+6:pos[0]+26,pos[1]+6:pos[1]+26,:]
			
			if get_dev(res)>(40/255):
				rem=comp(proc_gray(res))
				rem[0]=rem[0]*100
				rem[1]=rem[1]*100
				rem[2:]=(0.25+rem[2:])*40
				rem=np.floor(rem)
				rem=rem.astype(int)
				lep=out[pos[0]+15,pos[1]+15,j-1]
				lepo=int(np.floor(lep*100))
				if (np.logical_or(rem<0,rem>20).sum()==0):
					bi=np.ravel_multi_index(tuple(rem),(21,21,21,21,21,21))
					fi=np.argwhere(indx==bi)

					#print(tuple(rem))
					#print(bi)
					#print(fi)
					#pdb.set_trace()
					#print(fi.size)		
					if bi in indx:
						print("Found a patch in bin: ",tuple(rem))
						fold="{0:0=4d}".format(top_bin-fi[0][0])
						fold+="_"
						fold+=str(bi)
						edges+=1
						dist[top_bin-1-int(fi[0]),lepo]+=1
						if(not os.path.isdir(os.path.join(base,"tro",fold))):
							os.mkdir(os.path.join(base,"tro",fold))
							print(fold)
						namestr=''
						namestr+="{0:0=4d}".format(int(np.floor(1000*lep)))
						namestr+='_'
						namestr+="{0:0=6d}".format(len(os.listdir(os.path.join(base,"tro",fold))))
						name=os.path.join(base,'tro',fold,namestr)
						save_img(name+".png",resc)			
						plt.imsave(name+"_bw.png",res,cmap='gray')
						print(bi,name)
				else:
					lost_edges+=1
			else:
				subt_edges+=1

	print("Found: ",edges)
	print("Subt: ",subt_edges)
	print("Lost: ",lost_edges)
	print("Total: ",cnt)
	print("Saved after ",i," images")
	
	np.save('dist_save.npy',dist)

	
		
pdb.set_trace()


img=load_pyr(os.path.join(base,'lots','pyr_lep_data','pyr_lep (5).mat'))
pdb.set_trace()
#res=get_patch('pyr_img.png',img,0.4,0.7)
#plt.imshow(img,cmap='Greys',interpolation='none')
#plt.imsave('pyr_lep_test.png',img,cmap='Greys')

#plt.imshow(res[2])
#plt.show()
#plt.imshow(img[res[0]-20:res[0]+20,res[1]-20:res[1]+20],cmap='Greys',interpolation='none')
#plt.show()
'''



