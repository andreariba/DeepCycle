import argparse

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
from scipy import interpolate
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns

import math
import os
import shutil
import sys
from copy import copy
import time
import multiprocessing as mp
import json

import anndata

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='Run DeepCycle.')
parser.add_argument('--input_adata',type=str, required=True,help='Anndata input file preprocessed with velocyto and scvelo (moments).')
parser.add_argument('--gene_list',type=str, required=True,help='Subset of genes to run the inference on.')
parser.add_argument('--base_gene',type=str, required=True,help='Gene used to have an initial guess of the phase.')
parser.add_argument('--expression_threshold',type=float, required=True,help='Unsplced/spliced expression threshold.')
parser.add_argument('--gpu',const=True,default=False, nargs='?',help='Use GPUs.')
parser.add_argument('--hotelling',const=True,default=False, nargs='?',help='Use Hotelling filter.')
parser.add_argument('--output_adata',type=str, required=True,help='Anndata output file. ')
args = parser.parse_args()

#REQUIRED INPUTS
input_adata_file = args.input_adata
list_of_genes_file = args.gene_list
output_anndata_file = args.output_adata
base_gene = args.base_gene
GPU = args.gpu
HOTELLING = args.hotelling
expression_threshold = args.expression_threshold

cwd = os.getcwd()
print("[Current working directory]:",cwd)

CycleAE_dir = cwd+'/DeepCycle'
if not os.path.exists( CycleAE_dir ):
    os.makedirs( CycleAE_dir )

#TRAINING SETTINGS
fraction_of_cells_to_validation = 0.17
BATCH_SIZE = 5       # number of data points in each batch
N_EPOCHS = 1000           # times to run the model on complete data
lr = 1e-4          # initial learning rate

print("[TensorFlow version]:", tf.version.VERSION)

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

if GPU:
    print("[Using GPUs]")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices,'GPU')
    print("[Number of available GPUs]:", len(tf.config.experimental.list_physical_devices('GPU')))


class Hotelling:
    
    def __init__(self, x, y, dt, eta):
        self.x = x
        self.y = y
        self.sd_x = np.std(x)
        self.sd_y = np.std(y)
        self.delta_t = dt
        self.eta = eta
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        s = 100000
        self.hx = (self.xmax-self.xmin)/s
        self.hy = (self.ymax-self.ymin)/s
        self.flag = 'OK'
        try:
            self.density, self.potential, self.X, self.Y, self.Z = self.potential_estimation()
            self.x_ss_min, self.x_ss_max = self.find_minima()
            hessian_min = self.hessian_estimation(self.x_ss_min)
            hessian_max = self.hessian_estimation(self.x_ss_max)

            self.covariance_min = np.linalg.inv(hessian_min)
            self.covariance_max = np.linalg.inv(hessian_max)
            pooled_inverse_covariance_matrix = np.linalg.inv(0.5*(self.covariance_max+self.covariance_min))

            if self.confidence_level(self.x_ss_max)<0.05 or self.confidence_level(self.x_ss_min)<0.05:
                self.hotelling_t_squared = 0
                self.flag = 'LOW_DENSITY'
            else:
                self.hotelling_t_squared = (self.x_ss_max - self.x_ss_min).dot(pooled_inverse_covariance_matrix.dot(self.x_ss_max - self.x_ss_min))
                if np.linalg.norm(self.x_ss_min-self.x_ss_max)<0.1:
                    self.hotelling_t_squared = 0
                    self.flag = 'TOO_CLOSE'
        except np.linalg.LinAlgError as err:
            if ( ('singular matrix' in str(err)) or ('Singular matrix' in str(err)) ):
                self.hotelling_t_squared = 0
                self.flag = 'SINGULAR_MATRIX'
            else:
                raise
        self.path = None

    def potential_estimation(self):
        
        X, Y = np.mgrid[self.xmin:self.xmax:200j, self.ymin:self.ymax:200j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([self.x, self.y])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        return kernel, kernel.logpdf, X, Y, Z
    
    def calculate_potential_force(self,path):
        
        #Initialization
        Fv = []
        hx = self.hx
        hy = self.hy
        potential = self.potential
        
        #Calculate force for each point in path
        for i in range(len(path)):
            x = path[i][0]
            y = path[i][1]
            dV_dx = ( potential((x+hx,y)) - potential((x-hx,y)) )/(2*hx)
            dV_dy = ( potential((x,y+hy)) - potential((x,y-hy)) )/(2*hy)
            orthogonal_potential = np.array([dV_dx,dV_dy]).flatten()
            Fv.append(orthogonal_potential)
        
        return Fv
    
    def find_minima(self):
        
        #Initialization
        potential = self.potential
        delta_t = self.delta_t
        eta = self.eta
        
        start_time = time.time()
        
        #2 minima first estimation with Gaussian Mixture Model
        xy_dict = {'x':self.x,'y':self.y}
        xy_df = pd.DataFrame(xy_dict)
        gm = GaussianMixture(n_components=2)
        gm.fit(xy_df)
        points = gm.means_
        
        #Refining the minima estimation with the potential descent
        velocities = [np.array([0.0,0.0]) for i in range(len(points))]
        for i in range(1000):
            potential_force = self.calculate_potential_force( points )
            acceleration = [potential_force[k]-eta*velocities[k] for k in range(len(potential_force)) ]
            old_points = copy(points)
            total_acceleration = 0.0
            for j in range(0,len(points)):
                points[j] += 0.5*acceleration[j]*delta_t**2 + velocities[j]*delta_t
                velocities[j] += acceleration[j]*delta_t
                total_acceleration += np.linalg.norm(acceleration[j])
            if np.linalg.norm(points[0]-points[1])<0.1 or total_acceleration < 0.00001:
                break
        
        x0 , x1 = points
        
        #Sort the minima
        if np.linalg.norm(x1) < np.linalg.norm(x0):
            return x1, x0
        else:
            return x0, x1
    
    def hessian_estimation(self,xp):
        
        #Estimate the inverse of the hessian matrix in x
        x = xp[0]
        y = xp[1]
        hx = self.hx
        hy = self.hy
        potential = self.potential
        d2V_dx2 = float( (potential((x+hx,y))-2*potential((x,y))+potential((x-hx,y)))/(hx**2) )
        d2V_dxdy = float( (potential((x+hx,y+hy))-potential((x+hx,y))-potential((x,y+hy))+potential((x,y)))/(hx*hy) )
        d2V_dy2 = float( (potential((x,y+hy))-2*potential((x,y))+potential((x,y+hy)))/(hy**2) )
        hessian = -np.array([[d2V_dx2,d2V_dxdy],[d2V_dxdy,d2V_dy2]])
        
        return hessian
    
    def plot_density(self,file=None):
        
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax

        data_xy = np.array([self.x,self.y])
        cov_matrix = np.cov( data_xy )
        v, w = np.linalg.eigh( cov_matrix )

        x_avg, y_avg = np.mean( data_xy, axis=1 )
        first_component = np.sqrt(v[1])*w[:,1]
        second_component = np.sqrt(v[0])*w[:,0]
        u = [first_component[0],second_component[0]]
        v = [first_component[1],second_component[1]]
        origin = [x_avg], [y_avg]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.rot90(self.Z), cmap=plt.cm.jet_r, extent=[xmin, xmax, ymin, ymax], aspect='auto',zorder=0)
        cset = ax.contour(self.X,self.Y,self.Z,zorder=1)
        ax.clabel(cset, inline=1, fontsize=10)
        ax.plot(self.x_ss_min[0],self.x_ss_min[1], 'k.', markersize=10, color='red')
        ax.plot(self.x_ss_max[0],self.x_ss_max[1], 'k.', markersize=10, color='red')
        ax.quiver(x_avg, y_avg, first_component[0], first_component[1], color='white', angles='xy', scale_units='xy', scale=1,zorder=3)
        ax.quiver(x_avg, y_avg, second_component[0], second_component[0], color='gray', angles='xy', scale_units='xy', scale=1,zorder=3)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title('T-squared='+str(self.hotelling_t_squared))
        ax.set_xlabel('spliced')
        ax.set_ylabel('unspliced')
        if file!=None:
            plt.savefig(file)
    
    def confidence_level(self, point):
        density = self.density
        iso = density(point)
        sample = density.resample(size=200)
        insample = density(sample)<iso
        integral = insample.sum() / float(insample.shape[0])
        return integral



#CIRCULAR LAYER FOR THE AUTOENCODER
class Circularize(tfkl.Layer):
    def __init__(self, input_dim=1):
        super(Circularize, self).__init__()

    def call(self, inputs):
        return tf.concat([tf.math.cos(2*np.pi*inputs),tf.math.sin(2*np.pi*inputs)], axis=1)


sys.stdout.write("[Loading anndata]: ")
sys.stdout.flush()
adata = anndata.read_h5ad(input_adata_file)
print(input_adata_file)


sys.stdout.write("[Loading genes]: ")
cell_cycle_genes = []
with open(list_of_genes_file,'r') as fp:
    for line in fp:
        line = line.rstrip()
        if line != '':
            cell_cycle_genes.append(line)
print(list_of_genes_file)

#FILTER GENES BY HOTELLING
gene_list = list(adata.var.index)

hot_dir = CycleAE_dir+'/hotelling/'

def process_gene(gene, adata=adata, hot_dir=hot_dir):
    try:
        n = gene_list.index(gene)
        hotelling = 0
        df = pd.DataFrame({ 'spliced':adata.layers['Ms'][:,n], 'unspliced':adata.layers['Mu'][:,n] })
        if (df.mean()<0.5).all():
            return
        hot = Hotelling( x=df['spliced'], y=df['unspliced'], dt=0.1, eta=10 )
        if hot.flag!='OK':
            lock.acquire()
            print('['+gene+']: discarded because of', hot.flag)
            lock.release()
            return
        hotelling = abs(hot.hotelling_t_squared)
    except ValueError:
        lock.acquire()
        print('['+gene+']: discarded because of ValueError')
        lock.release()
        return
    if hotelling<0.5:
        lock.acquire()
        print('['+gene+']: discarded because of low T-squared')
        lock.release()
        return
    else:
        svgfile_hotelling = hot_dir+gene+'.svg'
        try:
            hot.plot_density(svgfile_hotelling)
        except Exception as e:
            lock.acquire()
            print('['+gene+']: error saving',svgfile_hotelling, e)
            lock.release()
        lock.acquire()
        print('['+gene+']: OK')
        lock.release()
        return(gene)

if HOTELLING:
    try:
        if os.path.exists(hot_dir):
            shutil.rmtree(hot_dir, ignore_errors=True)
            os.mkdir(hot_dir)
        else:
            os.mkdir(hot_dir)
    except:
        print("[ERROR]: Creation of the directory %s failed" % hot_dir)
        raise
        
    def init(l):
        global lock
        lock = l
    
    l = mp.Lock()
    pool = mp.Pool(initializer=init, initargs=(l,))
    pool_output = pool.map(process_gene, cell_cycle_genes)
    
    filtered_cell_cycle_genes = [i for i in pool_output if i]
    
    gene_json_file = hot_dir+'filtered_genes.json'
    with open(gene_json_file,'w') as fp:
        json.dump(filtered_cell_cycle_genes, fp)
    
    print("[Filtered genes]:", gene_json_file)
else:
    filtered_cell_cycle_genes = cell_cycle_genes

#FUNCTION TO GENERATE THE INPUTS OF THE AUTOENCODER
def generate_input(list_of_genes, adata):
    gene_list = list(adata.var.index)
    
    n = gene_list.index(list_of_genes[0])
    df_all = pd.DataFrame({ 'spliced':adata.layers['Ms'][:,n], 'unspliced':adata.layers['Mu'][:,n] })

    for gene in list_of_genes[1:]:
        try:
            n = gene_list.index(gene)
            df = pd.DataFrame({ 'spliced':adata.layers['Ms'][:,n], 'unspliced':adata.layers['Mu'][:,n] })
            if (df.mean()<expression_threshold).all():
                list_of_genes.remove(gene)
                continue
        except ValueError:
            list_of_genes.remove(gene)
            continue
        df_all = pd.concat([df_all, df], axis=1)

    normalized_df=(df_all-df_all.mean())/df_all.std()
    np_data = normalized_df.to_numpy()
    nan_columns = list(np.where(np.any(~np.isfinite(np_data),axis=0))[0])
    nan_indexes = list([ n//2 for n in nan_columns])
    even_columns = list([ 2*n for n in nan_indexes ])
    odd_columns = list([ 2*n+1 for n in nan_indexes ])
    for idx in sorted(nan_indexes, reverse=True):
        del list_of_genes[idx]
    columns_to_drop = sorted(even_columns + odd_columns)
    normalized_df.columns = list(range(normalized_df.shape[1]))
    df_all.columns = list(range(df_all.shape[1]))
    df_all = df_all.drop(df_all.columns[columns_to_drop], axis=1)
    normalized_df = normalized_df.drop(normalized_df.columns[columns_to_drop], axis=1)
    #print("Left genes: ",len(list_of_genes), normalized_df.shape)
    
    return list_of_genes, normalized_df.to_numpy() #cells x (2 genes)

#GENERATE INPUTS
genes, np_data = generate_input(filtered_cell_cycle_genes, adata)
print("[Genes]:", genes)
print("[N. OF USED GENES]",len(genes))
np.savez( CycleAE_dir+"/input_data.npz", genes, np_data, allow_pickle=True)
n_genes = len(genes)
n_cells = np_data.shape[0]
n_columns = np_data.shape[1]
if n_columns != 2*n_genes:
    print("[ERROR]: incoherent number of genes and columns")

n_cells_to_validation = int(n_cells*fraction_of_cells_to_validation)
print( "[Total number of cells]:", n_cells)
print( "[Number of cells used for training]:", n_cells-n_cells_to_validation)
print( "[Number of cells used for validation]:", n_cells_to_validation)


#BUILD INITIAL GUESS BASED ON A GENE 
index_gene = genes.index(base_gene)

angles = 1.5*(np.array([ [math.atan2(np_data[i,2*index_gene+1],np_data[i,2*index_gene])] for i in range(len(np_data)) ]) % (2*np.pi))/(2*np.pi)


#PARAMETERS
INPUT_DIM = len(genes)*2     # size of each input
HIDDEN_DIM = len(genes)*4        # hidden dimension
LATENT_DIM = 1        # latent vector dimension

input_shape = (INPUT_DIM,)
encoded_size = LATENT_DIM
base_depth = HIDDEN_DIM
print("[Model input shape]:", input_shape)

#BUILD THE INPUT DATASET OBJECTS
e_dataset = tf.data.Dataset.from_tensor_slices((np_data,angles)).shuffle(1000).batch(BATCH_SIZE)
n_batches_to_validation = math.ceil(tf.data.experimental.cardinality(e_dataset).numpy() * fraction_of_cells_to_validation)
eval_e_dataset = e_dataset.take(n_batches_to_validation)
train_e_dataset = e_dataset.skip(n_batches_to_validation)

ae_dataset = tf.data.Dataset.from_tensor_slices((np_data,np_data)).shuffle(1000).batch(BATCH_SIZE)
eval_ae_dataset = ae_dataset.take(n_batches_to_validation)
train_ae_dataset = ae_dataset.skip(n_batches_to_validation)

#FUNCTION TO PLOT THE TRAINING
def plot_training(fit):
    best_epoch = fit.epoch[fit.history['val_loss'].index(min(fit.history['val_loss']))]
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(fit.epoch,fit.history['val_loss'],'.-',color='red', label='validation')
    ax.plot(fit.epoch,fit.history['loss'],'.-',color='orange', label='train')
    ax.set_yscale('log')
    ax.set(ylabel='MSE')
    ax.axvspan(best_epoch-0.5,best_epoch+0.5, alpha=0.5, color='red')
    ax.legend()
    print("[Best epoch]:", best_epoch)
    print("[MSE]:", min(fit.history['val_loss']))



#ENCODER
inputs = tfk.Input(shape=input_shape)

norm_atans = tfkl.Reshape((1,))((tf.math.atan2(inputs[...,2*index_gene+1],inputs[...,2*index_gene]) % (2*np.pi))/(2*np.pi) )

x = tfkl.GaussianNoise(0.01)(inputs)
x = tfkl.Dense(base_depth,activation=tf.nn.leaky_relu)(x)
x = tfkl.Dense(base_depth,activation=tf.nn.leaky_relu)(x)
x = tfkl.Dense(base_depth,activation=tf.nn.leaky_relu)(x)
x = tfkl.Dense(base_depth,activation=tf.nn.leaky_relu)(x)

noisy_atans = tfkl.GaussianNoise(0.05)(norm_atans)
#manip_inputs = tfkl.concatenate([x,noisy_angle], axis=1)

manip_inputs = tfkl.concatenate([x,norm_atans], axis=1)

manip_inputs = tfkl.Dense(1)(manip_inputs)

outputs = tfkl.GaussianNoise(0.05)(manip_inputs)

encoder = tfk.Model(inputs, outputs, name="non_seq_encoder_noisy")
#encoder.summary()

#DECODER
decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=encoded_size),
    Circularize(),
    tfkl.GaussianNoise(0.03),
    tfkl.Dense(base_depth, activation=tf.nn.leaky_relu),
    tfkl.Dense(base_depth, activation=tf.nn.leaky_relu),
    tfkl.Dense(base_depth, activation=tf.nn.leaky_relu),
    tfkl.Dense(base_depth, activation=tf.nn.leaky_relu),
    tfkl.Dense(input_shape[0],
               activation=None),
])

#decoder.summary()

#AUTOENCODER
ae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs))


#ENCODER PRETRAINING
encoder.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.mean_squared_error, metrics=[tf.keras.metrics.kullback_leibler_divergence])

e_fit = encoder.fit(train_e_dataset,
            epochs=N_EPOCHS,
            validation_data=eval_e_dataset, batch_size=BATCH_SIZE,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.00001),
                       tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=20, verbose=1, mode='auto', restore_best_weights=True)
                      ]
            )


if not os.path.exists(CycleAE_dir+'/training'):
    os.makedirs(CycleAE_dir+'/training')
print("[ ENCODER PRE-TRAINING ]")
plot_training(e_fit)
plt.savefig(CycleAE_dir+'/training/encoder_pretraining.svg')
plt.clf()

#AUTOENCODER TRAINING
ae.compile(optimizer=tf.optimizers.Adam(learning_rate=lr/10),
            loss=tf.keras.losses.mean_squared_error, metrics=[tf.keras.metrics.kullback_leibler_divergence])

ae_fit = ae.fit(train_ae_dataset,
            epochs=N_EPOCHS,
            validation_data=eval_ae_dataset, batch_size=BATCH_SIZE,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.000001),
                       tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0, patience=20, verbose=1, mode='auto', restore_best_weights=True)]
          )


print("[ AUTOENCODER TRAINING ]")
plot_training(ae_fit)
plt.savefig(CycleAE_dir+'/training/autoencoder_training.svg')
plt.clf()

# Save the entire model
if not os.path.exists( CycleAE_dir+'/model' ):
    os.makedirs( CycleAE_dir+'/model' )
ae.save( CycleAE_dir+'/model' )



#CALCULATE PHASES
predicted_dataset = ae.predict(ae_dataset)
test_input_to_decoder = tf.convert_to_tensor(np.array([np.arange(0,1,step=0.01)]).T )
predicted_decoder_test = decoder(test_input_to_decoder)

distances = distance_matrix(np_data,predicted_decoder_test)

t_closest_theta = tf.convert_to_tensor(np.array([]))
for cell in distances:
    dist = list(cell)
    t2 = test_input_to_decoder[dist.index(min(dist))]
    t_closest_theta = tf.concat([t_closest_theta, t2],0)

t_closest_theta = t_closest_theta*-1+1
adata.obs['cell_cycle_theta'] = t_closest_theta
adata.write_h5ad(output_anndata_file)
print("[Output anndata]:", output_anndata_file)


print("[PRELIMINARY ANALYSIS]")
th=adata.obs['cell_cycle_theta']

minima = min(th)
maxima = max(th)

norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)

plt.scatter(adata.obsm['X_umap'][:,0],adata.obsm['X_umap'][:,1],c=th,cmap=cm.coolwarm,alpha=0.5)
plt.colorbar(mapper)
plt.savefig(CycleAE_dir+'/umap_cell_cycle_theta.svg')
plt.clf()

flipped_time = np.flip(test_input_to_decoder[:,0])
good_genes = genes
sorted_good_genes = sorted(good_genes)

w = 3
h = (len(sorted_good_genes))//w +1
if (len(sorted_good_genes)) % w >0:
    h += 1

fit_dir = CycleAE_dir+'/fits/'

try:
    if os.path.exists(fit_dir):
        shutil.rmtree(fit_dir, ignore_errors=True)
        os.mkdir(fit_dir)
    else:
        os.mkdir(fit_dir)
except:
    print("[ERROR]: Creation of the directory %s failed" % fit_dir)
    raise


for gene in sorted_good_genes:
    index = genes.index(gene)
    fig, ax = plt.subplots()
    sns.scatterplot(np_data[:,2*index],np_data[:,2*index+1],alpha=0.1,ax=ax)
    sns.kdeplot(np_data[:,2*index],np_data[:,2*index+1],shade=False,cmap="Reds",ax=ax)
    ax.plot(predicted_dataset[:,2*index],predicted_dataset[:,2*index+1],'.',alpha=0.05,color='black')
    ax.set_title(genes[index])
    ax.set_xlabel('spliced (z-score)')
    ax.set_ylabel('unspliced (z-score)')
    ax.set_aspect(1)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(fit_dir+gene+'.svg')
    plt.clf()
    

for gene in sorted_good_genes:
    index = genes.index(gene)
    fig, ax = plt.subplots()
    ax.plot(flipped_time,predicted_decoder_test[:,2*index],'.-',alpha=0.5,color='red')
    ax.plot(flipped_time,predicted_decoder_test[:,2*index+1],'.-',alpha=0.5,color='green')
    ax.set_title(genes[index])
    ax.set_xlabel('transcriptional phase')
    ax.set_ylabel('expression (z-score)')
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(fit_dir+gene+'_series.svg')
    plt.clf()

 
