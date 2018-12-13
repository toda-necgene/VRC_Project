import numpy as np
import scipy
import scipy.signal
import pyworld.pyworld as pw
import importlib

def wave2world(x,fs):
    # if gpu_acceralation:
    #     cp=importlib.import_module('cupy')
    # else:
    #     cp=numpy
    f0 = dio(x,fs)
    sp = get_spectram_envelove(x,f0,fs)
    ap = get_aperiodicity(x,f0,fs)
    return f0,sp,ap
def world2wave(f0,sp,ap,fs):
    wave = None
    return wave

log_2=0.69314
epsilon=1e-12
def dio(x:np.array,
        fs:int,
        period:int=5,
        f0_floor:float=71.0,
        f0_ceil:float=800.0,
        ch_in_oct:int=2,
        speed:int= 16000,
        allowed_range:float=0.1) -> np.array:
    # initialization of constants
    n_bands = int(1+np.ceil(np.log(f0_ceil/f0_floor))/log_2*ch_in_oct)
    boundary_f0 = f0_floor * np.power(2.0,np.arange(1,n_bands+1)/ch_in_oct)
    # normalization
    decimation_ratio=  np.clip(speed,1,12)
    y_length = 1+len(x)//decimation_ratio
    actual_fs = fs/decimation_ratio
    rs=y_length + (4 * int( 1 + actual_fs/boundary_f0[0] / 2))
    fft_size = int(np.power(2,int(np.log(rs)/log_2)+1))
    # decimation(downsampling) to fft
    if decimation_ratio!=1:
        y=scipy.signal.decimate(x,decimation_ratio,ftype="iir")
    else:
        y=x
    mean_y=np.mean(y)
    y-=mean_y
    y[y_length:]=0.0
    y=scipy.fftpack.rfft(y,n=fft_size)
    y=np.array([y,-y])
    y[0,-1]=y[0,1]
    y[1,0]=0.0
    y[1,-1]=0.0
    cut_off= int(np.round(actual_fs/50)* 2 + 1)
    low_cut_filter=np.zeros([fft_size])
    low_cut_filter[:cut_off]=0.5-0.5*np.cos(np.arange(1,cut_off+1)*2*np.pi/(cut_off+1))
    sum_of_aptitude = np.sum(low_cut_filter[:cut_off])
    low_cut_filter[:cut_off] = low_cut_filter[:cut_off]/sum_of_aptitude
    low_cut_filter[-((cut_off-1)//2)+1:]=low_cut_filter[:-((cut_off-1)//2):-1]
    low_cut_filter[:cut_off]=low_cut_filter[(cut_off-1)//2:(cut_off-1)//2+cut_off]
    low_cut_filter[0]+=1.0
    
    y_filter=scipy.fftpack.rfft(low_cut_filter,n=fft_size)
    y_filter=np.array([y_filter,-y_filter])
    y_filter[0,-1]=y_filter[0,1]
    y_filter[1,0]=0.0
    y_filter[1,-1]=0.0

    y_spectrum=np.zeros([fft_size,2])
    y_spectrum[:,0]=y[0]*y_filter[0]-y[1]*y_filter[1]
    y_spectrum[:,1]=y[0]*y_filter[1]+y[1]*y_filter[0]
    f0_length=int(1000*len(x)/fs/period)+1
    f0_candidates=np.zeros([n_bands,f0_length])
    f0_scores=np.zeros([n_bands,f0_length])
    tmp_positions=np.arange(f0_length) * period / 1000
    # f0 estimation
    # getting filtered signal
    
    for j in range(n_bands):
        
        y_length=fft_size
        nega_i_l = np.zeros([y_length])
        nega_i   = np.zeros([y_length])
        posi_i_l = np.zeros([y_length])
        posi_i   = np.zeros([y_length])
        peak_i_l = np.zeros([y_length])
        peak_i   = np.zeros([y_length])
        dips_i_l = np.zeros([y_length])
        dips_i   = np.zeros([y_length])
        
        half_range_length=np.ceil(fs/boundary_f0[j]/2.0)
        low_pass_filter=np.zeros([fft_size]) 
        tmp=np.arange(half_range_length*4)/((half_range_length*4-1)*2*np.pi)
        low_pass_filter[:int(half_range_length*4)]=0.355768-0.487396*np.cos(tmp)+0.144232*np.cos(2*tmp)-0.012604*np.cos(4*tmp)
        low_pass_filter_spec_tmp=scipy.fftpack.rfft(low_pass_filter,n=fft_size)
        low_pass_filter_spec_tmp=[low_pass_filter_spec_tmp,-low_pass_filter_spec_tmp]
        low_pass_filter_spec_tmp[0][-1]=low_pass_filter_spec_tmp[0][1]
        low_pass_filter_spec_tmp[1][0]=0.0
        low_pass_filter_spec_tmp[1][-1]=0.0
        low_pass_filter_spec=np.zeros([fft_size,2])
        half_fft_size=fft_size//2
        ## convolution
        low_pass_filter_spec[:half_fft_size,0]=y_spectrum[:half_fft_size,0]*low_pass_filter_spec_tmp[0][:half_fft_size] \
                                                -y_spectrum[:half_fft_size,1]*low_pass_filter_spec_tmp[1][:half_fft_size]
        low_pass_filter_spec[:half_fft_size,1]=y_spectrum[:half_fft_size,0]*low_pass_filter_spec_tmp[1][:half_fft_size] \
                                                +y_spectrum[:half_fft_size,1]*low_pass_filter_spec_tmp[0][:half_fft_size]
        low_pass_filter_spec[half_fft_size:]=low_pass_filter_spec[:half_fft_size-1:-1]
        
        p_in=np.concatenate([low_pass_filter_spec[:,0],low_pass_filter_spec[:,1]])
        filtered_signal=scipy.fftpack.irfft(p_in,n=fft_size,axis=-1)*2
        
        # Conpensation of the deray
        # index_bias=int(half_range_length*2)
        # filtered_signal[:y_length]=filtered_signal[index_bias:index_bias+y_length]

        # F0 CandidateCounter
        zc_number_of_negatives=zero_crossing_engine(filtered_signal,fft_size,actual_fs,nega_i_l,nega_i)
        zc_number_of_positives=zero_crossing_engine(-filtered_signal,fft_size,actual_fs,posi_i_l,posi_i)
        delta = filtered_signal[:-1] - filtered_signal[1:]
        zc_number_of_peaks=zero_crossing_engine(delta,fft_size-1,actual_fs,peak_i_l,peak_i)
        zc_number_of_dips=zero_crossing_engine(-delta,fft_size-1,actual_fs,dips_i_l,dips_i)
        interpolated_f0_set=np.zeros([4,f0_length])
        r=(zc_number_of_negatives-2>0)*(zc_number_of_positives-2>0)*(zc_number_of_peaks-2>0)*(zc_number_of_dips-2>0) == 0
        if r:
            f0_scores[j,:]=np.full([y_length],100000)
            f0_candidates[j,:]=np.zeros([y_length])
            continue
        interpolated_f0_set[0]=interp1(nega_i_l,nega_i,zc_number_of_negatives,tmp_positions,f0_length)
        interpolated_f0_set[1]=interp1(posi_i_l,posi_i,zc_number_of_positives,tmp_positions,f0_length)
        interpolated_f0_set[2]=interp1(peak_i_l,peak_i,zc_number_of_peaks,tmp_positions,f0_length)
        interpolated_f0_set[3]=interp1(dips_i_l,dips_i,zc_number_of_dips,tmp_positions,f0_length)
        
        f0_candidates[j]=(interpolated_f0_set[0]+interpolated_f0_set[1]+interpolated_f0_set[2]+interpolated_f0_set[3])/4
        f0_scores[j]=np.linalg.norm(interpolated_f0_set[:]-f0_candidates[j],ord=2,axis=0)/1.73205081 # 1.73205081 -> sqrt(3)
        for i in range(f0_length):
            if f0_candidates[j,i] > boundary_f0[j] or f0_candidates[j,i] < boundary_f0[j] / 2 or f0_candidates[j,i] > f0_ceil or f0_candidates[j,i] <f0_floor:
                f0_scores[j,i]=100000
                f0_candidates[j,i]=0 
        
    f0_scores=f0_scores/(f0_candidates+epsilon)
    # getting best f0 counter
    c=0
    best_f0_counter=np.zeros([f0_scores.shape[1]])
    for n in np.argmin(f0_scores,axis=0):
        best_f0_counter[c] = f0_candidates[n,c]
        c+=1
    # fix f0 counter with 4 steps
    voice_range_minimum=int(0.5+1000/period/f0_floor)
    # step1
    f0_tmp1=np.zeros([f0_length])
    f0_base=np.zeros([f0_length])
    f0_base[voice_range_minimum:-voice_range_minimum]=best_f0_counter[voice_range_minimum:-voice_range_minimum]
    for i in range(voice_range_minimum,f0_length):
        f0_tmp1[i]= f0_base[i] if abs((f0_base[i]-f0_base[i-1])/(epsilon+f0_base[i]))<allowed_range else 0
    # step2
    # f0_tmp2=f0_tmp1.copy()
    center=(voice_range_minimum-1)//2
    for i in range(center,f0_length-center):
        for j in range(-center,center):
            if f0_tmp1[i+j]==0:
                f0_tmp1[i]=0
                break
    positive_index=np.zeros([f0_length])
    negative_index=np.zeros([f0_length])
    pcount=0
    ncount=0
    for i in range(1,f0_length):
        if f0_tmp1[i]==0 and f0_tmp1[i-1]!=0:
            negative_index[ncount]=i-1
            ncount+=1
        elif f0_tmp1[i]!=0 and f0_tmp1[i-1]==0:
            positive_index[pcount]=i
            pcount+=1
    # step3
    for i in range(ncount):
        limit = f0_length-1 if i==ncount-1 else negative_index[i+1]
        for j in range(negative_index[i],limit):
            f0_tmp1[j+1]=select_best_f0(f0_tmp1[j],f0_tmp1[j-1],f0_candidates,n_bands,j+1,allowed_range)
            if f0_tmp1[j+1]==0:
                break
    # step4
    for i in range(pcount-1,0,-1):
        limit = 1 if i==0 else positive_index[i-1]
        for j in range(positive_index[i],limit,-1):
            best_f0_counter[j-1]=select_best_f0(f0_tmp1[j],f0_tmp1[j+1],f0_candidates,n_bands,j-1,allowed_range)
            if best_f0_counter[j-1]==0:
                break
    
    return best_f0_counter
def select_best_f0(cur_f0,past_f0,f0_candidates,number_of_candidates,target_index,allowed_range):
    # ここから
    ref_f0=(cur_f0*3.0-past_f0)/2.0
    min_err=np.abs(ref_f0-f0_candidates[0][target_index])
    best_f0=f0_candidates[0][target_index]
    for  i in range(1,number_of_candidates):
        cur_err=np.abs(ref_f0-f0_candidates[i][target_index])
        if cur_err<min_err:
            min_err=cur_err
            best_f0=f0_candidates[i][target_index]
    if np.abs(1.0-best_f0/ref_f0)>allowed_range:
        return 0.0
    return best_f0
def interp1(x,y,xl,xi,xil):
    h=x[1:]-x[:-1]
    s=np.zeros([xil],dtype=np.int16)
    k=np.zeros([xil],dtype=np.int16)
    
    # histc
    i=0
    stage=0
    count=1
    while i < xil:
        if stage==0:
            k[i]=1
            if xi[i] >= x[0]:
                stage=1
        elif stage==1:
            if xi[i] < x[count]:
                k[i]=count
            else:
                k[i] = count
                i-=1
                count+=1           
            if count==xl:
                break
        i+=1
    count-=1
    i+=1
    while i < xil:
        k[i]=count
        i+=1
    #end histc
    yi=np.zeros([xil])
    for i in range(xil):
        s=(xi[i]-x[k[i]-1]) / h[k[i]-1]
        yi[i]=y[k[i]-1]+s*(y[k[i]]-y[k[i]-1])
    return yi
def zero_crossing_engine(signal,y_length,actual_fs,interval_locations,intervals):
    negative_going_points=np.zeros([y_length])
    for i in range(y_length-1):
        negative_going_points[i]= i+1 if 0.0 < signal[i] and signal[i+1] <= 0 else 0
    negative_going_points[-1]=0
    edges=np.zeros([y_length],dtype=np.int16)
    counts=0
    for i in range(y_length):
        if negative_going_points[i]>0:
            counts+=1
            edges[counts]=negative_going_points[i]
    if counts<=2:
        return 0
    fine_edges=np.zeros([counts])
    for i in range(counts):
        fine_edges[i]=edges[i]-signal[edges[i]-1]/(signal[edges[i]]-signal[edges[i]-1])
    for i in range(counts-1):
        intervals[i]=actual_fs/(fine_edges[i+1]-fine_edges[i])
        interval_locations[i]=(fine_edges[i]+fine_edges[i+1]) /2.0 / actual_fs
    return counts-1

import wave
import matplotlib.pyplot as plt
if __name__=="__main__":
    CHUNK=1024
    dms=[]
    wf = wave.open("dataset/test/test.wav", 'rb')
    dds = wf.readframes(CHUNK)
    while dds != b'':
        dms.append(dds)
        dds = wf.readframes(CHUNK)
    dms = b''.join(dms)
    data = np.frombuffer(dms, 'int16')
    data_real=(data/32767).reshape(-1).astype(np.float)
    data_realA=dmn=data_real.copy()
    f0,t=pw.dio(data_realA,16000)
    f0_alpha=dio(data_realA,16000)
    plt.plot(f0)
    plt.plot(f0_alpha)
    plt.show()