package com.example.newdemo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.lang.Math;

import org.opencv.android.JavaCameraView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Rect;

import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
//import org.opencv.video.BackgroundSubtractorMOG;

//import android.R;
import com.example.newdemo.R;
import com.ipaulpro.afilechooser.utils.FileUtils;
//import com.ipaulpro.afilechooserexample.FileChooserExampleActivity;

import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
//import android.provider.MediaStore;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
//import android.database.Cursor;
import android.hardware.Camera;
import android.hardware.Camera.AutoFocusCallback;
import android.support.v4.view.MotionEventCompat;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.Button;
import android.widget.Toast;
//import android.hardware.Camera.Size;

public class MainActivity extends Activity implements CvCameraViewListener2{

	private static final String TAG = "HandGestureApp";
	private static final int COLOR_SPACE = Imgproc.COLOR_RGB2Lab;
	private static final int GES_FRAME_MAX= 10;
	
	public final Object sync = new Object();
	
	public static final int SAMPLE_MODE = 0;
	public static final int DETECTION_MODE = 1;
	public static final int TRAIN_REC_MODE = 2;
	public static final int BACKGROUND_MODE = 3;
	public static final int ADD_MODE = 4;
	public static final int TEST_MODE = 5;
	
	private static final int FRAME_BUFFER_NUM = 1;
	private int testFrameCount = 0;
	private float[][] values = new float[FRAME_BUFFER_NUM][];
	private int[][] indices = new int[FRAME_BUFFER_NUM][];
	
	private static final int REQUEST_CODE = 6384; // onActivityResult request
      // code
	private String diagResult = null;
	private Handler mHandler = new Handler();
	private static final String DATASET_NAME = "/train_data.txt";
	
	private String storeFolderName = null;
	private File storeFolder = null;
	private FileWriter fw = null;
	
	private MyCameraView mOpenCvCameraView;
	private List<android.hardware.Camera.Size> mResolutionList;
	
	private int mode = BACKGROUND_MODE;
	
	private static final int SAMPLE_NUM = 7;
	
	
	private Point[][] samplePoints = null;
	private double[][] avgColor = null;
	private double[][] avgBackColor = null;
	
	private double[] channelsPixel = new double[4];
	private ArrayList<ArrayList<Double>> averChans = new ArrayList<ArrayList<Double>>();
	
	private double[][] cLower = new double[SAMPLE_NUM][3];
	private double[][] cUpper = new double[SAMPLE_NUM][3];
	private double[][] cBackLower = new double[SAMPLE_NUM][3];
	private double[][] cBackUpper = new double[SAMPLE_NUM][3];
	
	private Scalar lowerBound = new Scalar(0, 0, 0);
	private Scalar upperBound = new Scalar(0, 0, 0);
	private int squareLen;
	
	//private BackgroundSubtractorMOG mog;
	private Mat sampleColorMat = null;
	private List<Mat> sampleColorMats = null;
	
	private Mat[] sampleMats = null ;
	
	private Mat rgbaMat = null;
	
	private Mat rgbMat = null;
	private Mat bgrMat = null;
	
	
	private Mat interMat = null;
	//private Mat ret;
	private Mat binMat = null;
	private Mat binTmpMat = null;
	private Mat binTmpMat2 = null;
	private Mat binTmpMat0 = null;
	private Mat binTmpMat3 = null;
	
	private Mat tmpMat = null;
	private Mat backMat = null;
	private Mat difMat = null;
	private Mat binDifMat = null;
	
	/*private MatOfInt             mChannels[] = null;
    private MatOfInt             mHistSize = null;
    private int                  mHistSizeNum = 25;
    private MatOfFloat           mRanges = null;
    private Point                mP1 = null;
    private Point                mP2 = null;
    private float                mBuff[] = null;
    private Mat                  mMat0 = null;
    private Mat[]                hists = null;*/
    
    private Scalar               mColorsRGB[] = null;
    
	private HandGesture hg = null;
	
	private int imgNum;
	private int gesFrameCount;
	private int curLabel = 0;
	private int selectedLabel = -2;
	private int curMaxLabel = 0;
	
	private ArrayList<String> feaStrs = new ArrayList<String>();
	
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch(status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i("Android Tutorial", "OopenCV loaded successfully");
				
				System.loadLibrary("HandGestureApp");
				
				  try {
			            System.loadLibrary("signal");
			        } catch (UnsatisfiedLinkError ule) {
			            Log.e(TAG, "Hey, could not load native library signal");
			        }
				  
				mOpenCvCameraView.enableView();
				
				 mOpenCvCameraView.setOnTouchListener(new OnTouchListener() {
	    	    	    public boolean onTouch(View v, MotionEvent event) {
	    	    	        // ... Respond to touch events 
	    	    	    	 int action = MotionEventCompat.getActionMasked(event);
	    	    	         
	    	    	    	    switch(action) {
	    	    	    	        case (MotionEvent.ACTION_DOWN) :
	    	    	    	            Log.d(TAG,"Action was DOWN");
	    	    	    	            String toastStr = null;
	    	    	    	            if (mode == SAMPLE_MODE) {
	    	    	    	            	mode = DETECTION_MODE;
	    	    	    	            	toastStr = "Sampling Finished!";
	    	    	    	            } else if (mode == DETECTION_MODE) {
	    	    	    	            	mode = TRAIN_REC_MODE;
	    	    	    	            	((Button)findViewById(R.id.AddBtn)).setVisibility(View.VISIBLE);
	    	    	    	            	((Button)findViewById(R.id.TrainBtn)).setVisibility(View.VISIBLE);
	    	    	    	            	((Button)findViewById(R.id.TestBtn)).setVisibility(View.VISIBLE);
	    	    	    	            	toastStr = "Binary Display Finished!";
	    	    	    	            	
	    	    	    	            	preTrain();
	    	    	    	            	
	    	    	    	            } else if (mode == TRAIN_REC_MODE){
	    	    	    	            	mode = DETECTION_MODE;
	    	    	    	            	((Button)findViewById(R.id.AddBtn)).setVisibility(View.INVISIBLE);
	    	    	    	            	((Button)findViewById(R.id.TrainBtn)).setVisibility(View.INVISIBLE);
	    	    	    	            	((Button)findViewById(R.id.TestBtn)).setVisibility(View.INVISIBLE);
	    	    	    	            	
	    	    	    	            	toastStr = "train finished!";
	    	    	    	            } else if (mode == BACKGROUND_MODE) {
	    	    	    	            	toastStr = "First background sampled!";
	    	    	    	            	rgbaMat.copyTo(backMat);
	    	    	    	            	mode = SAMPLE_MODE;
	    	    	    	            }
	    	    	    	            
	    	    	    	        	Toast.makeText(getApplicationContext(), toastStr, Toast.LENGTH_LONG).show();
	    	    	    	            return false;
	    	    	    	        case (MotionEvent.ACTION_MOVE) :
	    	    	    	            Log.d(TAG,"Action was MOVE");
	    	    	    	            return true;
	    	    	    	        case (MotionEvent.ACTION_UP) :
	    	    	    	            Log.d(TAG,"Action was UP");
	    	    	    	            return true;
	    	    	    	        case (MotionEvent.ACTION_CANCEL) :
	    	    	    	            Log.d(TAG,"Action was CANCEL");
	    	    	    	            return true;
	    	    	    	        case (MotionEvent.ACTION_OUTSIDE) :
	    	    	    	            Log.d(TAG,"Movement occurred outside bounds " +
	    	    	    	                    "of current screen element");
	    	    	    	            return true;      
	    	    	    	        default : 
	    	    	    	            return true;
	    	    	    	    }      
	    	    	    	    
	    	    	        
	    	    	    }
	    	     });
				 
			} break;
			default: {
				super.onManagerConnected(status);
				
			}break;
			}
		}
	};
	
	 // svm native
    private native int trainClassifierNative(String trainingFile, int kernelType,
    		int cost, float gamma, int isProb, String modelFile);
    private native int doClassificationNative(float values[][], int indices[][],
    		int isProb, String modelFile, int labels[], double probs[]);
   
    private void train() {
    	// Svm training
    	int kernelType = 2; // Radial basis function
    	int cost = 4; // Cost
    	int isProb = 0;
    	float gamma = 0.001f; // Gamma
    	String trainingFileLoc = storeFolderName+DATASET_NAME;
    	String modelFileLoc = storeFolderName+"/model";
    	Log.i("Store Path", modelFileLoc);
    	
    	if (trainClassifierNative(trainingFileLoc, kernelType, cost, gamma, isProb,
    			modelFileLoc) == -1) {
    		Log.d(TAG, "training err");
    		finish();
    	}
    	Toast.makeText(this, "Training is done", 2000).show();
    }
    
	public void initLabel() {
		//String path = Environment.getExternalStorageDirectory().toString()+"/";
		//Log.d("Files", "Path: " + path);
		       
		File file[] = storeFolder.listFiles();
	//	Log.d("Files", "Size: "+ file.length);
		int maxLabel = 0;
		for (int i=0; i < file.length; i++)
		{
		  //  Log.d("Files", "FileName:" + file[i].getName());
			String fullName = file[i].getName();
			
			final int dotId = fullName.lastIndexOf('.');
			if (dotId > 0) {
				String name = fullName.substring(0, dotId);
				String extName = fullName.substring(dotId+1);
				if (extName.equals("jpg")) {
					int curName = Integer.valueOf(name);
					if (curName > maxLabel)
						maxLabel = curName;
				}
				
			}
		}
		
		curLabel = maxLabel;
		curMaxLabel = curLabel;
		
	}
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		mOpenCvCameraView = (MyCameraView) findViewById(R.id.HandGestureApp);
		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
		mOpenCvCameraView.setCvCameraViewListener(this);
		
		samplePoints = new Point[SAMPLE_NUM][2];
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				samplePoints[i][j] = new Point();
			}
		}
		
		avgColor = new double[SAMPLE_NUM][3];
		avgBackColor = new double[SAMPLE_NUM][3];
		
		for (int i = 0; i < 3; i++)
			averChans.add(new ArrayList<Double>());
		
		//HLS
		//initCLowerUpper(7, 7, 80, 80, 80, 80);
		
		//RGB
		//initCLowerUpper(30, 30, 30, 30, 30, 30);
		
		//HSV
		//initCLowerUpper(15, 15, 50, 50, 50, 50);
		//initCBackLowerUpper(5, 5, 80, 80, 100, 100);
		
		//Ycrcb
	//	initCLowerUpper(40, 40, 10, 10, 10, 10);
		
		//Lab
		initCLowerUpper(50, 50, 10, 10, 10, 10);
	    initCBackLowerUpper(50, 50, 3, 3, 3, 3);
		
		SharedPreferences numbers = getSharedPreferences("Numbers", 0);
        imgNum = numbers.getInt("imgNum", 0);
		
        
        
        initOpenCV();
        
        Log.i(TAG, "Created!");
	}

	public void initOpenCV() {
		
	}
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
	    // Handle item selection
	    switch (item.getItemId()) {
	        case R.id.action_save:
	            savePicture();
	            return true;
	        case R.id.data_collection:
	        	callDataCollection();
	        	return true;
	        	
	        default:
	            return super.onOptionsItemSelected(item);
	    }
	}
	
	public void showDialogBeforeAdd(String title,String message){
		Log.i("Show Dialog", "Entered");
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(  
                  this);  
             // set title  
             alertDialogBuilder.setTitle(title);  
             // set dialog message  
             alertDialogBuilder  
                  .setMessage(message)  
                  .setCancelable(false)  
                  .setPositiveButton("Yes",new DialogInterface.OnClickListener() {  
                       public void onClick(DialogInterface dialog,int id) {  
                           
                            doAddNewGesture();
                            
                            synchronized(sync) {
                            	sync.notify();
                            }
                            
                            dialog.cancel();  
                            
                           
                       }  
                   })  
                  .setNegativeButton("No",new DialogInterface.OnClickListener() {  
                       public void onClick(DialogInterface dialog,int id) {  
                            // if this button is clicked, just close  
                            // the dialog box and do nothing 
                    	   
                    	   synchronized(sync) {
                               sync.notify();
                               }
                    	   
                            dialog.cancel(); 
                            
                           
                       }  
                  });  
                  // create alert dialog  
                  AlertDialog alertDialog = alertDialogBuilder.create();  
                  // show it  
                  alertDialog.show();  
   }  
	
	public void showDialog(final Context v, String title,String message, 
			String posStr, String negStr, String neuStr){
		//Log.i("Show Dialog", "Entered");
		
		diagResult = null;
		
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(  
                  v);  
             // set title  
             alertDialogBuilder.setTitle(title);  
             // set dialog message  
             alertDialogBuilder  
                  .setMessage(message)  
                  .setCancelable(false)  
                  .setPositiveButton(posStr,new DialogInterface.OnClickListener() {  
                       public void onClick(DialogInterface dialog,int id) {  
                           
                           
                            diagResult = "Positive";
                            
                            
                            Toast.makeText(getApplicationContext(), "Add more to Gesture "
         							+ selectedLabel, Toast.LENGTH_SHORT).show();
                            
                            curLabel = selectedLabel - 1;
                            
                            
                            dialog.cancel();  
                            
                           
                       }  
                   })  
                  .setNegativeButton(negStr,new DialogInterface.OnClickListener() {  
                       public void onClick(DialogInterface dialog,int id) {  
                            // if this button is clicked, just close  
                            // the dialog box and do nothing 
                    	   
                    	  
                    	    diagResult = "Negative"; 
                    	    
                    	    doDeleteGesture(selectedLabel);
                    	    
                    	    Toast.makeText(getApplicationContext(), "Gesture "
         							+ selectedLabel + " is deleted", Toast.LENGTH_SHORT).show();
                    	            	             	    
                    	    curLabel = selectedLabel - 1;
                            dialog.cancel(); 
                            
                           
                       }  
                  });  
             
             if (neuStr != null) {
             alertDialogBuilder.setNeutralButton(neuStr, new DialogInterface.OnClickListener() {  
                 public void onClick(DialogInterface dialog,int id) {  
                     
                     
                     diagResult = "Neutral";
                     
                     Toast.makeText(getApplicationContext(), "Canceled"
  							, Toast.LENGTH_SHORT).show();
                     
                     selectedLabel = -2;
                     dialog.cancel();  
                     
                    
                }  
             })  ;
             }
                  // create alert dialog  
                  AlertDialog alertDialog = alertDialogBuilder.create();  
                  // show it  
                  alertDialog.show();  
   }  
	
	/* Checks if external storage is available for read and write */
 	public boolean isExternalStorageWritable() {
 	    String state = Environment.getExternalStorageState();
 	    if (Environment.MEDIA_MOUNTED.equals(state)) {
 	        return true;
 	    }
 	    return false;
 	}
 	
 	public void callDataCollection() {
 		if (mode != TRAIN_REC_MODE) {
 			Toast.makeText(getApplicationContext(), "Please do it in training mode!", Toast.LENGTH_SHORT).show();
 		} else {
 			String dataPath;
 			if (storeFolder != null) {
 				dataPath = storeFolderName;
 			} else {
 				dataPath = Environment.getExternalStorageDirectory().toString();
 			}
 			
 		//	Intent intent = new Intent();  
 		
 		//	intent.setAction(Intent.ACTION_GET_CONTENT); 
 		//	intent.setType("*/*");
 			 
 		//	startActivityForResult(Intent.createChooser(intent, "Choose A Gesture"), 1);
 			
 			selectedLabel = -2;
 			showChooser();
 			
 			
 		}
 	}
 	
 	 private void showChooser() {
         // Use the GET_CONTENT intent from the utility class
         Intent target = FileUtils.createGetContentIntent();
         // Create the chooser Intent
         Intent intent = Intent.createChooser(
                 target, getString(R.string.chooser_title));
         try {
             startActivityForResult(intent, REQUEST_CODE);
         } catch (ActivityNotFoundException e) {
             // The reason for the existence of aFileChooser
         }
     }

     @Override
     protected void onActivityResult(int requestCode, int resultCode, Intent data) {
         switch (requestCode) {
             case REQUEST_CODE:
                 // If the file selection was successful
                 if (resultCode == RESULT_OK) {
                     if (data != null) {
                         // Get the URI of the selected file
                         final Uri uri = data.getData();
                         Log.i(TAG, "Uri = " + uri.toString());
                         try {
                             // Get the file path from the URI
                             final String path = FileUtils.getPath(this, uri);
                           //  Toast.makeText(this,
                            //         "File Selected: " + path, Toast.LENGTH_LONG).show();
                             
                             int slashId = path.lastIndexOf('/');
                             int slashDot = path.lastIndexOf('.');
                             String selectedLabelStr = path.substring(slashId+1, slashDot);
                             selectedLabel = Integer.valueOf(selectedLabelStr);
                             
                             if (selectedLabel != -2) {
                  				showDialog(this, "Add or Delete", "Selected Label is " + 
                  			selectedLabel + ",\nAdd to this gesture or delete it?", 
                  			"Add", "Delete", "Cancel");
                  			
                  			}
                             
                         } catch (Exception e) {
                             Log.e("FileSelectorTestActivity", "File select error", e);
                         }
                     }
                 }
                 break;
         }
         super.onActivityResult(requestCode, resultCode, data);
     }
 	
 	public void addNewGesture(View view) {
 	    // Do something in response to button click
 		
 		if (mode == TRAIN_REC_MODE) {
 		if (storeFolder != null) {
 			File myFile = new File(storeFolderName + DATASET_NAME);
			
 			if (myFile.exists()) {
 				
 			} else {
 				try {
 				myFile.createNewFile();
 				} catch (Exception e) {
 					Toast.makeText(getApplicationContext(), "Failed to create dataset at "
 							+ myFile, Toast.LENGTH_SHORT).show();
 				}
 			}
 			
 			
			 try {
				 fw = new FileWriter(myFile, true);
				// fw.write("Hello World\n");
				// fw.flush();
				 
				 feaStrs.clear();
				// curLabel++;
				 
				 if (selectedLabel == -2)
					 curLabel = curMaxLabel + 1;
				 else {
					 curLabel++;
					 selectedLabel = -2;
				 }
				 
				 
				 gesFrameCount = 0;
				 mode = ADD_MODE;
			     
				 
		    } catch (FileNotFoundException e) {
		        e.printStackTrace();
		        Log.i(TAG, "******* File not found. Did you" +
		                " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
		    } catch (IOException e) {
		        e.printStackTrace();
		    }   
 		}
 		} else {
 			Toast.makeText(getApplicationContext(), "Please do it in TRAIN_REC mode"
					, Toast.LENGTH_SHORT).show();
 		}
 		
 	}
 	
 	public void doAddNewGesture() {
 		try {
 		for (int i = 0; i < feaStrs.size(); i++) {
			fw.write(feaStrs.get(i));
		}
 		
 		fw.close();
 		
 		} catch (Exception e) {
 			
 		}
 		
 		savePicture();
 		
 		if (curLabel > curMaxLabel) {
 			curMaxLabel = curLabel;
 		}
 		
		
 	}
 	
 	public void doDeleteGesture(int label) {
 		
 	}
 	
 	public void preTrain() { //Open or create training data set
 		 if (!isExternalStorageWritable()) {
 			 Toast.makeText(getApplicationContext(), "External storage is not writable!", Toast.LENGTH_SHORT).show();
		 } else if (storeFolder == null) {
 			storeFolderName = Environment.getExternalStorageDirectory() + "/MyDataSet";
 			storeFolder = new File(storeFolderName);
 			boolean success = true;
 			if (!storeFolder.exists()) {
 			    success = storeFolder.mkdir();
 			}
 			if (success) {
 			    // Do something on success
 				
 				
 			} else {
 				Toast.makeText(getApplicationContext(), "Failed to create directory "+ storeFolderName, Toast.LENGTH_SHORT).show();
 				storeFolder = null;
 				storeFolderName = null;
 			}
 		}
 		 
 		 if (storeFolder != null) {
 			 initLabel();
 		 }
 		 
 		
 	}
 	
 	public void train(View view) {
 		train();
 	}
 	
 	public void test(View view) {
 		
 		if (mode == TRAIN_REC_MODE)
 			mode = TEST_MODE;
 		else if (mode == TEST_MODE) {
 			mode = TRAIN_REC_MODE;
 		}
 	}
 	
	boolean savePicture()
	{
		Mat img;
		
		
		
		if (((mode == BACKGROUND_MODE) || (mode == SAMPLE_MODE)
				|| (mode == TRAIN_REC_MODE)) || (mode == ADD_MODE)) {
			Imgproc.cvtColor(rgbaMat, bgrMat, Imgproc.COLOR_RGBA2BGR, 3);
			img = bgrMat;
		} else if (mode == DETECTION_MODE) {
			img = binMat;
		} else 
			img = null;
		
		if (img != null) {
			 if (!isExternalStorageWritable()) {
	 	//		 Toast.makeText(getApplicationContext(), "External storage is not writable!", Toast.LENGTH_SHORT).show();
				  return false;
			 }
	 	
			 File path;
			 String filename;
			 if (mode != ADD_MODE) {
				 path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
				 filename = "image_" + imgNum + ".jpg";
			 } else {
				 path = storeFolder;
				 filename = curLabel + ".jpg";
			 }
			 
			 
			 imgNum++;
			 File file = new File(path, filename);
		
			 
			  
			  Boolean bool = false;
			  filename = file.toString();
			  
			
			  bool = Highgui.imwrite(filename, img);
		
			  if (bool == true) {
				//  Toast.makeText(getApplicationContext(), "Saved as " + filename, Toast.LENGTH_SHORT).show();
				  Log.d(TAG, "Succeed writing image to" + filename);
			  } else
			    Log.d(TAG, "Fail writing image to external storage");
			  
			  return bool;
		}
		
		return false;
	}
	
	//Just initialize boundaries of the first sample
	void initCLowerUpper(double cl1, double cu1, double cl2, double cu2, double cl3,
			double cu3)
	{
		cLower[0][0] = cl1;
		cUpper[0][0] = cu1;
		cLower[0][1] = cl2;
		cUpper[0][1] = cu2;
		cLower[0][2] = cl3;
		cUpper[0][2] = cu3;
	}
	
	void initCBackLowerUpper(double cl1, double cu1, double cl2, double cu2, double cl3,
			double cu3)
	{
		cBackLower[0][0] = cl1;
		cBackUpper[0][0] = cu1;
		cBackLower[0][1] = cl2;
		cBackUpper[0][1] = cu2;
		cBackLower[0][2] = cl3;
		cBackUpper[0][2] = cu3;
	}
	
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	 AutoFocusCallback myAutoFocusCallback = new AutoFocusCallback(){

		  @Override
		  public void onAutoFocus(boolean arg0, Camera arg1) {
		   // TODO Auto-generated method stub
			  int maxFocusAreas = mOpenCvCameraView.getMaxNumFocusAreas();
			  
			  Log.i("MaxFocusAreas", "" + maxFocusAreas);
		  }};
		  
	public void checkCameraParameters()
	{
		/*if (mOpenCvCameraView.isAutoWhiteBalanceLockSupported()) {
			if (mOpenCvCameraView.getAutoWhiteBalanceLock()) {
				Log.d("AutoWhiteBalanceLock", "Locked");
			} else {
				Log.d("AutoWhiteBalanceLock", "Not Locked");
				mOpenCvCameraView.setAutoWhiteBalanceLock(true);
				
				if (mOpenCvCameraView.getAutoWhiteBalanceLock()) {
					Log.d("AutoWhiteBalanceLockAfter", "Locked");
				}
			}
		} else {
			Log.d("AutoWhiteBalanceLock", "Not Supported");
		}*/
		Log.d("Focus Mode", mOpenCvCameraView.getFocusMode());
		mOpenCvCameraView.setFocusModeCon();
		Log.d("Focus Mode After", mOpenCvCameraView.getFocusMode());
		mOpenCvCameraView.startAutoFocus(myAutoFocusCallback);
		
		/*if (mOpenCvCameraView.isAutoExposureLockSupported()) {
			if (mOpenCvCameraView.getAutoExposureLock()) {
				Log.d("AutoExposureLock", "Locked");
			} else {
				Log.d("AutoExposureLock", "Not Locked");
				mOpenCvCameraView.setAutoExposureLock(true);
				
				if (mOpenCvCameraView.getAutoExposureLock()) {
					Log.d("AutoExposureLockAfter", "Locked");
				}
				
			}
		} else {
			Log.d("AutoExposureLock", "Not Supported");
		}*/
	}
	
	public void releaseCVMats() {
		releaseCVMat(sampleColorMat);
		
		if (sampleColorMats!=null) {
			for (int i = 0; i < sampleColorMats.size(); i++)
			{
				releaseCVMat(sampleColorMats.get(i));
			}
		}
		
		if (sampleMats != null) {
			for (int i = 0; i < sampleMats.length; i++)
			{
				releaseCVMat(sampleMats[i]);
			}
		}
		
		releaseCVMat(rgbMat);
		releaseCVMat(bgrMat);
		releaseCVMat(interMat);
		releaseCVMat(binMat);
		releaseCVMat(binTmpMat0);
		releaseCVMat(binTmpMat3);
		releaseCVMat(binTmpMat2);
		releaseCVMat(tmpMat);
		releaseCVMat(backMat);
		releaseCVMat(difMat);
		releaseCVMat(binDifMat);
	}
	
	public void releaseCVMat(Mat img) {
		if (img != null)
			img.release();

	}
	
	@Override
	public void onCameraViewStarted(int width, int height) {
		// TODO Auto-generated method stub
		Log.i(TAG, "On cameraview started!");
		
		mResolutionList = mOpenCvCameraView.getResolutionList();
		android.hardware.Camera.Size resolution;
		for (int i = 0; i < mResolutionList.size(); i++)
		{
			resolution = mResolutionList.get(i);
			int resWidth = resolution.width;
			int resHeight = resolution.height;
			
			if ((resWidth < 200) && (resHeight < 200)) {
				mOpenCvCameraView.setResolution(resolution);
				//String caption = Integer.valueOf(resWidth).toString() + "x" + Integer.valueOf(resHeight).toString();
				//Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
				
				Log.i("CurrentResolution", "width: "+resWidth+", height: "+resHeight);
				break;
			}
			
			
			
		}
		
		//checkCameraParameters();
		releaseCVMats();
		
		sampleColorMat = new Mat();
		
	
			
		sampleColorMats = new ArrayList<Mat>();
		
		sampleMats = new Mat[SAMPLE_NUM];
		for (int i = 0; i < SAMPLE_NUM; i++)
			sampleMats[i] = new Mat();
		
		rgbMat = new Mat();
		
		bgrMat = new Mat();
		
		interMat = new Mat();
		
		binMat = new Mat();
		
		binTmpMat = new Mat();
		
		binTmpMat2 = new Mat();
		
		binTmpMat0 = new Mat();
		
		binTmpMat3 = new Mat();
		
		tmpMat = new Mat();
		
		backMat = new Mat();
		
		
		difMat = new Mat();
		binDifMat = new Mat();
		
		
		hg = new HandGesture();
		
		
		
		/*mChannels = new MatOfInt[] { new MatOfInt(0), new MatOfInt(1), new MatOfInt(2) };
	    mBuff = new float[mHistSizeNum];
	    mHistSize = new MatOfInt(mHistSizeNum);
	    mRanges = new MatOfFloat(0f, 256f);
	    mMat0  = new Mat();
	    hists = new Mat[3];
	    for (int i = 0; i < 3; i++)
	    	hists[i] = new Mat();
	    
	    mP1 = new Point();
        mP2 = new Point();*/
	    
        mColorsRGB = new Scalar[] { new Scalar(255, 0, 0, 255), new Scalar(0, 255, 0, 255), new Scalar(0, 0, 255, 255) };
        
	}

	@Override
	public void onCameraViewStopped() {
		// TODO Auto-generated method stub
		Log.i(TAG, "On cameraview stopped!");
		//releaseCVMats();
	}

	@Override
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		// TODO Auto-generated method stub
		rgbaMat = inputFrame.rgba();
		//Mat ret = new Mat();
		Core.flip(rgbaMat, rgbaMat, 1);
		
		//android.hardware.Camera.Size curRes = mOpenCvCameraView.getResolution();
		//Log.d(TAG, "Resolution: " + curRes.width + " x " + curRes.height);
		
		Imgproc.GaussianBlur(rgbaMat, rgbaMat, new Size(5,5), 5, 5);
		
		Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB);
		
		Imgproc.cvtColor(rgbaMat, interMat, COLOR_SPACE);
		
		//subtractBack();
		
		if (mode == SAMPLE_MODE) {
			//Imgproc.cvtColor(rgbaMat, interMat, COLOR_SPACE);
			preSampleHand(rgbaMat);
			//rgba.copyTo(ret);
		//	return rgbaMat;
			//Imgproc.pyrDown(rgba, ret);
			//genHistogram(interMat);
		//	Log.d(TAG, ret.dump());
			//ret = interMat;
		} else if (mode == DETECTION_MODE) {
			//Imgproc.blur(rgbaMat, interMat, new Size(3, 3));
		//	Imgproc.GaussianBlur(rgbaMat, interMat, new Size(3,3), 1, 1);
			//return interMat;
			
		//	Imgproc.cvtColor(interMat, interMat, COLOR_SPACE);
			produceBinImg(interMat, binMat);
			
			
			return binMat;
		//	return binDifMat;
		
		} else if ((mode == TRAIN_REC_MODE)||(mode == ADD_MODE)
		|| (mode == TEST_MODE)){
		//	Imgproc.GaussianBlur(rgbaMat, interMat, new Size(3,3), 1, 1);
			//return interMat;
			
		//	Imgproc.cvtColor(interMat, interMat, COLOR_SPACE);
			produceBinImg(interMat, binMat);
			
			makeContours();
		
			
			
			String entry = hg.featureExtraction(rgbaMat, curLabel);
			
			if (mode == ADD_MODE) {
				gesFrameCount++;
				Core.putText(rgbaMat, Integer.toString(gesFrameCount), new Point(10, 
				10), Core.FONT_HERSHEY_SIMPLEX, 0.6, Scalar.all(0));
				
				
				
				feaStrs.add(entry);
				
				if (gesFrameCount == GES_FRAME_MAX) {
					
					 Runnable runnableShowBeforeAdd = new Runnable() {
				            @Override
				            public void run() {                
				                {                    
				                	showDialogBeforeAdd("Add or not", "Add this new gesture labeled as "
											+ curLabel + "?");
				                }
				            }
				        };     
				        
					mHandler.post(runnableShowBeforeAdd);
					
					
					
					try {
						
						synchronized(sync) {
							sync.wait();
						}
					
					
					
					
					
					} catch (Exception e) {
						
					}
					
					
					
					mode = TRAIN_REC_MODE;
				}
			} else if (mode == TEST_MODE) {
				Double[] doubleValue = hg.features.toArray(new Double[hg.features.size()]);
				values[0] = new float[doubleValue.length];
				indices[0] = new int[doubleValue.length];
				
				for (int i = 0; i < doubleValue.length; i++)
				{
					values[0][i] = (float)(doubleValue[i]*1.0f);
					indices[0][i] = i+1;
				}
				
				int isProb = 0;
				String modelFile = storeFolderName + "/model";
				int[] returnedLabel = {0};
				double[] returnedProb = {0.0};
				int r = doClassificationNative(values, indices, isProb, modelFile, returnedLabel, returnedProb);
				
				if (r == 0) {
					Core.putText(rgbaMat, Integer.toString(returnedLabel[0]), new Point(15, 
							15), Core.FONT_HERSHEY_SIMPLEX, 0.6, mColorsRGB[0]);
				}
			}
			
		//	return interMat;
		} else if (mode == BACKGROUND_MODE) {
			preSampleBack(rgbaMat);
		} 
		
		return rgbaMat;
	//	return interMat;
	
	}
	
	void preSampleHand(Mat img)
	{
		int cols = img.cols();
		int rows = img.rows();
		squareLen = rows/20;
		Scalar color = mColorsRGB[2];  //Blue Outline
		//Log.d(TAG, "cols: " + cols + ", rows: " + rows);
		
		
		samplePoints[0][0].x = cols/2;
		samplePoints[0][0].y = rows/4;
		samplePoints[1][0].x = cols*5/12;
		samplePoints[1][0].y = rows*5/12;
		samplePoints[2][0].x = cols*7/12;
		samplePoints[2][0].y = rows*5/12;
		samplePoints[3][0].x = cols/2;
		samplePoints[3][0].y = rows*7/12;
		samplePoints[4][0].x = cols/1.5;
		samplePoints[4][0].y = rows*7/12;
		samplePoints[5][0].x = cols*4/9;
		samplePoints[5][0].y = rows*3/4;
		samplePoints[6][0].x = cols*5/9;
		samplePoints[6][0].y = rows*3/4;
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			samplePoints[i][1].x = samplePoints[i][0].x+squareLen;
			samplePoints[i][1].y = samplePoints[i][0].y+squareLen;
		}
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			Core.rectangle(img,  samplePoints[i][0], samplePoints[i][1], color, 1);
		}
		
		//average(interMat);
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				avgColor[i][j] = (interMat.get((int)(samplePoints[i][0].y+squareLen/2), (int)(samplePoints[i][0].x+squareLen/2)))[j];
			}
		}
		
		
	}
	
	void preSampleBack(Mat img)
	{
		int cols = img.cols();
		int rows = img.rows();
		squareLen = rows/20;
		Scalar color = mColorsRGB[2];  //Blue Outline
		//Log.d(TAG, "cols: " + cols + ", rows: " + rows);
		
		
		samplePoints[0][0].x = cols/6;
		samplePoints[0][0].y = rows/3;
		samplePoints[1][0].x = cols/6;
		samplePoints[1][0].y = rows*2/3;
		samplePoints[2][0].x = cols/2;
		samplePoints[2][0].y = rows/6;
		samplePoints[3][0].x = cols/2;
		samplePoints[3][0].y = rows/2;
		samplePoints[4][0].x = cols/2;
		samplePoints[4][0].y = rows*5/6;
		samplePoints[5][0].x = cols*5/6;
		samplePoints[5][0].y = rows/3;
		samplePoints[6][0].x = cols*5/6;
		samplePoints[6][0].y = rows*2/3;
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			samplePoints[i][1].x = samplePoints[i][0].x+squareLen;
			samplePoints[i][1].y = samplePoints[i][0].y+squareLen;
		}
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			Core.rectangle(img,  samplePoints[i][0], samplePoints[i][1], color, 1);
		}
		
		//average(interMat);
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				avgBackColor[i][j] = (interMat.get((int)(samplePoints[i][0].y+squareLen/2), (int)(samplePoints[i][0].x+squareLen/2)))[j];
			}
		}
		
	}
	
	void average(Mat img)
	{
	
		for (int i = 0; i < 3; i++)
			averChans.get(i).clear();
		
		sampleColorMats.clear();
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			//subImg = img.submat((int)samplePoints[i][0].y+2, (int)samplePoints[i][1].y-2, 
			//		(int)samplePoints[i][0].x+2, (int)samplePoints[i][1].x-2);
			Mat roiSample = new Mat(img, new Rect((int)samplePoints[i][0].x+2, (int)samplePoints[i][0].y+2, squareLen-4, squareLen-4));
			Mat tmpSample = new Mat();
			roiSample.copyTo(tmpSample);
			sampleColorMats.add(tmpSample);
			
			for (int x = (int)samplePoints[i][0].x+2; x <= (int)samplePoints[i][1].x-2; x++)
			{
				for (int y = (int)samplePoints[i][0].y+2; y <= (int)samplePoints[i][1].y-2; y++)
				{
					channelsPixel = img.get(y, x);
					
					if ((x == (int)samplePoints[i][0].x+2)&&
							(y == (int)samplePoints[i][0].y+2))
						//Log.d(TAG, "ref " + x + ", " + y + ": " + channelsPixel[0] + ", "
						//		+ channelsPixel[1] + ", " + channelsPixel[2]);
					
					for (int j = 0; j < 3; j++)
					{
						averChans.get(j).add(Double.valueOf(channelsPixel[j]));
					}
				}
			}
			for (int j = 0; j < 3; j++)
		    {
				Collections.sort(averChans.get(j));
				int len = averChans.get(j).size();
				
				
				if (mode == BACKGROUND_MODE)
					avgBackColor[i][j] = averChans.get(j).get(len/2).doubleValue();
				else if (mode == SAMPLE_MODE)
					avgColor[i][j] = averChans.get(j).get(len/2).doubleValue();
				
				//Log.d(TAG, i + ", " + j + ":" + avgColor[i][j]);
		    }
		}
	}
	
	/*void genHistogram(Mat img)
	{
		Core.vconcat(sampleColorMats, sampleColorMat);
		int rows = img.rows();
		int cols = img.cols();
		
		
		int thikness = (int) ((cols / 3 - 5) / mHistSizeNum);
     
        int offset = 5;
        
		  for(int c=0; c<3; c++) {
		      Imgproc.calcHist(Arrays.asList(img), mChannels[c], mMat0, hists[c], mHistSize, mRanges);
		      Core.normalize(hists[c], hists[c], rows/2, 0, Core.NORM_INF);
		      hists[c].get(0, 0, mBuff);
		      for(int h=0; h<mHistSizeNum; h++) {
		         
		    	  mP1.x = mP2.x = offset + h*thikness;
		          mP1.y = rows-1;
		          mP2.y = mP1.y - 2 - (int)mBuff[h];
		          Core.line(rgbaMat, mP1, mP2, mColorsRGB[c], thikness);
		      }
		      
		      offset = (int)mP1.x + thikness + 5;
		  }
		
	}*/
	
	void boundariesCorrection()
	{
		for (int i = 1; i < SAMPLE_NUM; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cLower[i][j] = cLower[0][j];
				cUpper[i][j] = cUpper[0][j];
				
				cBackLower[i][j] = cBackLower[0][j];
				cBackUpper[i][j] = cBackUpper[0][j];
			}
		}
		
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (avgColor[i][j] - cLower[i][j] < 0)
					cLower[i][j] = avgColor[i][j];
				
				if (avgColor[i][j] + cUpper[i][j] > 255)
					cUpper[i][j] = 255 - avgColor[i][j];
				
				if (avgBackColor[i][j] - cBackLower[i][j] < 0)
					cBackLower[i][j] = avgBackColor[i][j];
				
				if (avgBackColor[i][j] + cBackUpper[i][j] > 255)
					cBackUpper[i][j] = 255 - avgBackColor[i][j];
			}
		}
	}
	
	
	
	void cropBinImg(Mat imgIn, Mat imgOut)
	{
		imgIn.copyTo(binTmpMat3);
		
		Rect boxRect = makeBoundingBox(binTmpMat3);
		Rect finalRect = null;
		
		if (boxRect!=null) {
		Mat roi = new Mat(imgIn, boxRect);
		int armMargin = 2;
		
		Point tl = boxRect.tl();
		Point br = boxRect.br();
		
		int colNum = imgIn.cols();
		int rowNum = imgIn.rows();
		
		int wristThresh = 10;
		
		List<Integer> countOnes = new ArrayList<Integer>();
		
		if (tl.x < armMargin) {
			double rowLimit = br.y;
			int localMinId = 0;
			for (int x = (int)tl.x; x < br.x; x++)
			{
				int curOnes = Core.countNonZero(roi.col(x));
				int lstTail = countOnes.size()-1;
				if (lstTail >= 0) {
					if (curOnes < countOnes.get(lstTail)) {
						localMinId = x;
					}
				}
				
				if (curOnes > (countOnes.get(localMinId) + wristThresh))
					break;
				
				countOnes.add(curOnes);
			}
			
			Rect newBoxRect = new Rect(new Point(localMinId, tl.y), br);
			roi = new Mat(imgIn, newBoxRect);
			
			Point newtl = newBoxRect.tl();
			Point newbr = newBoxRect.br();
			
			int y1 = (int)newBoxRect.tl().y;
			while (Core.countNonZero(roi.row(y1)) < 2) {
				y1++;
			}
			
			int y2 = (int)newBoxRect.br().y;
			while (Core.countNonZero(roi.row(y2)) < 2) {
				y2--;
			}
			finalRect = new Rect(new Point(newtl.x, y1), new Point(newbr.x, y2));
		} else if (br.y > rowNum - armMargin) {
			double rowLimit = br.y;
			
			
		/*	int localMinId = (int)br.y - 1;
			for (int y = (int)br.y - 1; y > tl.y; y--)
			{
				int curOnes = Core.countNonZero((roi.row(y - (int)tl.y)));
				int lstTail = countOnes.size()-1;
				if (lstTail >= 0) {
					if (curOnes < countOnes.get(lstTail)) {
						localMinId = y;
					}
					
					countOnes.add(curOnes);
					if (curOnes > (countOnes.get((int)br.y-1-localMinId) + wristThresh))
						break;
					
				} else
					countOnes.add(curOnes);
			}*/
			
			int scanCount = 0;
			int scanLength = 8;
			int scanDelta = 8;
			int y;
			for (y = (int)br.y - 1; y > tl.y; y--)
			{
				int curOnes = Core.countNonZero((roi.row(y - (int)tl.y)));
				int lstTail = countOnes.size()-1;
				if (lstTail >= 0) {
					countOnes.add(curOnes);
					
					if (scanCount % scanLength == 0) {
						int curDelta = curOnes - countOnes.get(scanCount-5);
						if (curDelta > scanDelta)
							break;
					}
					
					
				} else
					countOnes.add(curOnes);
				
				scanCount++;
			}
			
			
			
			
			finalRect = new Rect(tl, new Point(br.x, y+scanLength));
			
		/*	Rect newBoxRect = new Rect(tl, new Point(br.x, localMinId));
			roi = new Mat(imgIn, newBoxRect);
			
			Point newtl = newBoxRect.tl();
			Point newbr = newBoxRect.br();
			
			
			int x1 = (int)newBoxRect.tl().x;
			
			for (; x1 < (int)newbr.x; x1++) {
				if (Core.countNonZero(roi.col(x1-(int)newtl.x)) < 2)
					break;
			}
			
			
			int x2 = (int)newBoxRect.br().x - 1;
			for (; x2 > (int)newtl.x; x2--) 
			{
				if (Core.countNonZero(roi.col(x2 - (int)newtl.x)) < 2) {
					break;
				}
			}
			
			finalRect = new Rect(new Point(x1, newtl.y), new Point(x2, newbr.y));*/
			
		}
		

		if (finalRect!=null) {
			roi = new Mat(imgIn, finalRect);
			roi.copyTo(tmpMat);
			imgIn.copyTo(imgOut);
			imgOut.setTo(Scalar.all(0));
			roi = new Mat(imgOut, finalRect);
			tmpMat.copyTo(roi);
		}
		
		
		}
		
	}
	
	void adjustBoundingBox(Rect initRect, Mat img)
	{
		
	}
	
	void produceBinImg(Mat imgIn, Mat imgOut)
	{
		int colNum = imgIn.cols();
		int rowNum = imgIn.rows();
		int boxExtension = 0;
		
		boundariesCorrection();
		
		produceBinHandImg(imgIn, binTmpMat);
	
		
		//binTmpMat.copyTo(binTmpMat0);
		//binTmpMat.copyTo(imgOut);
		
		
		
		
		produceBinBackImg(imgIn, binTmpMat2);
	//	produceBinBackImg(imgIn, imgOut);
		
		Core.bitwise_and(binTmpMat, binTmpMat2, binTmpMat);
		binTmpMat.copyTo(tmpMat);
		binTmpMat.copyTo(imgOut);
		
		Rect roiRect = makeBoundingBox(tmpMat);
		adjustBoundingBox(roiRect, binTmpMat);
		
		if (roiRect!=null) {
			roiRect.x = Math.max(0, roiRect.x - boxExtension);
			roiRect.y = Math.max(0, roiRect.y - boxExtension);
			roiRect.width = Math.min(roiRect.width+boxExtension, colNum);
			roiRect.height = Math.min(roiRect.height+boxExtension, rowNum);
			
		//	Mat roi1 = new Mat(binTmpMat0, roiRect);	
		//	Mat roi2 = new Mat(binTmpMat2, roiRect);
			
			Mat roi1 = new Mat(binTmpMat, roiRect);
			Mat roi3 = new Mat(imgOut, roiRect);
			imgOut.setTo(Scalar.all(0));
			
			roi1.copyTo(roi3);
		//	binTmpMat2.copyTo(imgOut);
			
		//	Core.bitwise_or(roi1, roi2, roi3);
			
		//	imgOut.setTo(Scalar.all(0));
		//	roi1.copyTo(roi3);
		//	binTmpMat0.copyTo(imgOut);
			
			Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));	
			Imgproc.dilate(roi3, roi3, element, new Point(-1, -1), 2);
			
			Imgproc.erode(roi3, roi3, element, new Point(-1, -1), 2);
			
			//Imgproc.adaptiveBilateralFilter(imgOut, tmpMat, new Size(5, 5), 50);
			//tmpMat.copyTo(imgOut);
		}
		
	//	cropBinImg(imgOut, imgOut);
		
	}
	
	void produceBinHandImg(Mat imgIn, Mat imgOut)
	{
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
			lowerBound.set(new double[]{avgColor[i][0]-cLower[i][0], avgColor[i][1]-cLower[i][1],
					avgColor[i][2]-cLower[i][2]});
			upperBound.set(new double[]{avgColor[i][0]+cUpper[i][0], avgColor[i][1]+cUpper[i][1],
					avgColor[i][2]+cUpper[i][2]});
			
		
			
			
			Core.inRange(imgIn, lowerBound, upperBound, sampleMats[i]);
			
			
			
		//	Log.d("lowerBound ", i + ": " + lowerBound.toString());
		//	Log.d("upperBound ", i + ": " + upperBound.toString());
			
		}
		
		imgOut.release();
		sampleMats[0].copyTo(imgOut);
	//	imgOut = sampleMats[0];
		
		
		//Log.d(TAG, imgOut.dump());
		
		for (int i = 1; i < SAMPLE_NUM; i++)
		{
			Core.add(imgOut, sampleMats[i], imgOut);
		}
		
	//	subtractBack();
		
		//dilate(imgOut);
		
		
		blackCorners(imgOut);
		
		Imgproc.medianBlur(imgOut, imgOut, 3);
	}
	
	void produceBinBackImg(Mat imgIn, Mat imgOut)
	{
		for (int i = 0; i < SAMPLE_NUM; i++)
		{
		/*	lowerBound.set(new double[]{avgColor[i][0]-cLower[i][0], avgColor[i][1]-cLower[i][1],
					avgColor[i][2]-cLower[i][2]});
			upperBound.set(new double[]{avgColor[i][0]+cUpper[i][0], avgColor[i][1]+cUpper[i][1],
					avgColor[i][2]+cUpper[i][2]});*/
			
			lowerBound.set(new double[]{avgBackColor[i][0]-cBackLower[i][0], avgBackColor[i][1]-cBackLower[i][1],
					avgBackColor[i][2]-cBackLower[i][2]});
			upperBound.set(new double[]{avgBackColor[i][0]+cBackUpper[i][0], avgBackColor[i][1]+cBackUpper[i][1],
					avgBackColor[i][2]+cBackUpper[i][2]});
			
			
			Core.inRange(imgIn, lowerBound, upperBound, sampleMats[i]);
			//Core.bitwise_not(sampleMats[i], sampleMats[i]);
			
			
		//	Log.d("lowerBound ", i + ": " + lowerBound.toString());
		//	Log.d("upperBound ", i + ": " + upperBound.toString());
			
		}
		
		imgOut.release();
		sampleMats[0].copyTo(imgOut);
	//	imgOut = sampleMats[0];
		
		
		//Log.d(TAG, imgOut.dump());
		
		for (int i = 1; i < SAMPLE_NUM; i++)
		{
			Core.add(imgOut, sampleMats[i], imgOut);
		}
		
		Core.bitwise_not(imgOut, imgOut);
		
	//	subtractBack();
		
		//dilate(imgOut);
		
		
		blackCorners(imgOut);
		
		Imgproc.medianBlur(imgOut, imgOut, 7);
		
		
	}
	
	
	void dilate(Mat img)
	{
		int cols = img.cols();
		int rows = img.rows();
		
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
		
		Mat roi = new Mat(img, new Rect(cols/4, rows/6, cols/2, rows*2/3));
		
		
		Imgproc.dilate(roi, roi, element);
		
		//Imgproc.bilateralFilter(roi, tmpMat, 5, 50, 50);
		//Imgproc.adaptiveBilateralFilter(roi, tmpMat, new Size(5, 5), 50);
	//	tmpMat.copyTo(roi);
	}
	

	
	void makeContours()
	{
		hg.contours.clear();
		Imgproc.findContours(binMat, hg.contours, hg.hie, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
		hg.findBiggestContour();
		
		if (hg.cMaxId > -1) {
			//MatOfPoint2f tmpMat2f = new MatOfPoint2f();
			//tmpMat2f.fromList(hg.contours.get(hg.cMaxId).toList());
			//Imgproc.approxPolyDP(tmpMat2f, tmpMat2f, 2, true);
			//hg.contours.get(hg.cMaxId).fromList(tmpMat2f.toList());
			hg.approxContour.fromList(hg.contours.get(hg.cMaxId).toList());
			Imgproc.approxPolyDP(hg.approxContour, hg.approxContour, 2, true);
			hg.contours.get(hg.cMaxId).fromList(hg.approxContour.toList());
			
			Imgproc.drawContours(rgbaMat, hg.contours, hg.cMaxId, mColorsRGB[0], 1);
			
			hg.findInscribedCircle(rgbaMat);
			
			/*Moments mu = Imgproc.moments(hg.contours.get(hg.cMaxId));
			hg.momentCenter.x = mu.get_m10()/mu.get_m00();
			hg.momentCenter.y = mu.get_m01()/mu.get_m00();
			hg.calculateTilt(mu.get_m11(), mu.get_m20(), mu.get_m02());
			
			double k = Math.tan(hg.momentTiltAngle);
			
			double x_p = hg.momentCenter.x+5;
			double y_p = k*5 + hg.momentCenter.y;*/
			
			
		//	Core.circle(rgbaMat, hg.momentCenter, 2, mColorsRGB[0], -5);
			
		//	Core.line(rgbaMat, hg.momentCenter, new Point(x_p, y_p), mColorsRGB[0], 2);
			
			hg.boundingRect = Imgproc.boundingRect(hg.contours.get(hg.cMaxId));
			
			Imgproc.convexHull(hg.contours.get(hg.cMaxId), hg.hullI, false);
			
			hg.hullP.clear();
			for (int i = 0; i < hg.contours.size(); i++)
				hg.hullP.add(new MatOfPoint());
			
			int[] cId = hg.hullI.toArray();
			List<Point> lp = new ArrayList<Point>();
			Point[] contourPts = hg.contours.get(hg.cMaxId).toArray();
			
			for (int i = 0; i < cId.length; i++)
			{
				lp.add(contourPts[cId[i]]);
				//Core.circle(rgbaMat, contourPts[cId[i]], 2, new Scalar(241, 247, 45), -3);
			}
			hg.hullP.get(hg.cMaxId).fromList(lp);
			lp.clear();
			
			
			
			hg.fingerTips.clear();
			hg.defectPoints.clear();
			hg.defectPointsOrdered.clear();
			
			hg.fingerTipsOrdered.clear();
			hg.defectIdAfter.clear();
			
		//	hg.hullCurP.fromArray(hg.hullP.get(hg.cMaxId).toArray());
		//	Imgproc.approxPolyDP(hg.hullCurP, hg.approxHull, 5, true);
			
			if ((contourPts.length >= 5) 
					&& hg.detectIsHand(rgbaMat) && (cId.length >=5)){
				Imgproc.convexityDefects(hg.contours.get(hg.cMaxId), hg.hullI, hg.defects);
				List<Integer> dList = hg.defects.toList();
			
				//List<Integer> idList = new ArrayList<Integer>();
				//HashSet<Integer> idList = new HashSet<Integer>();
				
				/*int lastDefectIdx = -1;
				
				
				List<Point> defectList = new ArrayList<Point>();
				for (int i = 0; i < dList.size(); i++)
				{
					if (i % 4 == 2) {
						Point curPoint = (hg.contours.get(hg.cMaxId).toArray())[dList.get(i)];
						if ((curPoint.x > 3) && (curPoint.y > 3) && 
								(curPoint.x < rgbaMat.cols()-3) &&
								(curPoint.y < rgbaMat.rows()-3)) {
							defectList.add(curPoint);
						}
								
						
					}
				}
				hg.palmDefects.fromList(defectList);
				Moments muDefects = Imgproc.moments(hg.palmDefects);
				hg.palmCenter.x = muDefects.get_m10()/muDefects.get_m00();
				hg.palmCenter.y = muDefects.get_m01()/muDefects.get_m00();*/
				
				
			//	hg.palmCenter.x = hg.inCircle.x;
			//	hg.palmCenter.y = hg.inCircle.y;
				
				Point prevPoint = null;
				
				for (int i = 0; i < dList.size(); i++)
				{
					int id = i % 4;
					Point curPoint;
					
					if (id == 2) { //Defect point
						double depth = (double)dList.get(i+1)/256.0;
						curPoint = contourPts[dList.get(i)];
											
					//	Point disToCenter = new Point(curPoint.x-hg.palmCenter.x, curPoint.y-hg.palmCenter.y);
						
					//	double disCenter = Math.sqrt((disToCenter.x*disToCenter.x+disToCenter.y*disToCenter.y));
						
						Point curPoint0 = contourPts[dList.get(i-2)];
						Point curPoint1 = contourPts[dList.get(i-1)];
						Point vec0 = new Point(curPoint0.x - curPoint.x, curPoint0.y - curPoint.y);
						Point vec1 = new Point(curPoint1.x - curPoint.x, curPoint1.y - curPoint.y);
						double dot = vec0.x*vec1.x + vec0.y*vec1.y;
						double lenth0 = Math.sqrt(vec0.x*vec0.x + vec0.y*vec0.y);
						double lenth1 = Math.sqrt(vec1.x*vec1.x + vec1.y*vec1.y);
						double cosTheta = dot/(lenth0*lenth1);
						
						if ((depth > hg.inCircleRadius*0.7)&&(cosTheta>=-0.7)
								&& (!isClosedToBoundary(curPoint0, rgbaMat))
								&&(!isClosedToBoundary(curPoint1, rgbaMat))
								){
							
						/*	hg.defectPoints.add(curPoint);
							
							Point defectVec = new Point(curPoint.x-hg.palmCenter.x, 
									curPoint.y-hg.palmCenter.y);
							double defectAngle = Math.atan2(defectVec.y, defectVec.x);
							hg.defectPointsOrdered.put(defectAngle, i);
							
							idList.add(Integer.valueOf(dList.get(i-2)));
							
							idList.add(Integer.valueOf(dList.get(i-1)));*/
							
						//	Point curDefectVec = new Point(curPoint.x-hg.inCircle.x, curPoint.y-hg.inCircle.y);
							
							
							hg.defectIdAfter.add((i));
							
							
							Point finVec0 = new Point(curPoint0.x-hg.inCircle.x,
									curPoint0.y-hg.inCircle.y);
							double finAngle0 = Math.atan2(finVec0.y, finVec0.x);
							Point finVec1 = new Point(curPoint1.x-hg.inCircle.x,
									curPoint1.y - hg.inCircle.y);
							double finAngle1 = Math.atan2(finVec1.y, finVec1.x);
							
							
							
							if (hg.fingerTipsOrdered.size() == 0) {
								hg.fingerTipsOrdered.put(finAngle0, curPoint0);
								hg.fingerTipsOrdered.put(finAngle1, curPoint1);
								
							} else {
							   
							    //	Point dis = new Point(prevPoint.x-curPoint0.x, prevPoint.y-curPoint.y);
							    	
							   // 	if (dis.x*dis.x + dis.y*dis.y >= 400) {
							    		hg.fingerTipsOrdered.put(finAngle0, curPoint0);
							   // 	}
							    	
							    	hg.fingerTipsOrdered.put(finAngle1, curPoint1);
							    
							  
							}
							
							
						//	hg.fingerTips.add(curPoint0);
						//	lastDefectIdx = i;
						}
						
						else {
							
							//if ((!isClosedToBoundary(curPoint, rgbaMat))
							//		&& (depth >= 10)) {
							//	Core.circle(rgbaMat, curPoint, 2, mColorsRGB[1], -2);
							//	hg.defectPoints.add(curPoint);
								
						//	}
							
							
						}
									
					
					}
				}
				
				/*  Iterator it=idList.iterator();

		          while(it.hasNext())
		         {
		            int value = (Integer)it.next();
		            hg.fingerTips.add((hg.contours.get(hg.cMaxId).toArray())[value]);

		        
		         }*/
			        
				/*if (lastDefectIdx > -1) {
					Point curPoint1 = (hg.contours.get(hg.cMaxId).toArray())[dList.get(lastDefectIdx-1)];
					
					
					hg.fingerTips.add(curPoint1);
					
					
				}*/
			}
			
		/*	hg.fingerTips.clear();
			for (int i = 0; i < cId.length; i++)
			{
				hg.fingerTips.add((hg.contours.get(hg.cMaxId).toArray())[cId[i]]);
			}*/
			
			
		}
		
		if (hg.detectIsHand(rgbaMat)) {

			Core.rectangle(rgbaMat, hg.boundingRect.tl(), hg.boundingRect.br(), mColorsRGB[1], 2);
			Imgproc.drawContours(rgbaMat, hg.hullP, hg.cMaxId, mColorsRGB[2]);
		}
		
	}
	
	boolean isClosedToBoundary(Point pt, Mat img)
	{
		int margin = 5;
		if ((pt.x > margin) && (pt.y > margin) && 
				(pt.x < img.cols()-margin) &&
				(pt.y < img.rows()-margin)) {
			return false;
		}
		
		return true;
	}
	
	Rect makeBoundingBox(Mat img)
	{
		hg.contours.clear();
		Imgproc.findContours(img, hg.contours, hg.hie, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
		hg.findBiggestContour();
		
		if (hg.cMaxId > -1) {
			//Imgproc.drawContours(rgbaMat, hg.contours, hg.cMaxId, mColorsRGB[0], 1);
			
			hg.boundingRect = Imgproc.boundingRect(hg.contours.get(hg.cMaxId));
			
			
			
		}
		
		if (hg.detectIsHand(rgbaMat)) {

		//	Core.rectangle(rgbaMat, hg.boundingRect.tl(), hg.boundingRect.br(), mColorsRGB[1], 2);
			return hg.boundingRect;
		} else
			return null;
	}
	
	void blackCorners(Mat img)
	{
		int cols = img.cols();
		int rows = img.rows();
		
		
	}
	
	@Override
	public void onPause(){
		Log.i(TAG, "Paused!");
		super.onPause();
	/*	if (mOpenCvCameraView != null){
			mOpenCvCameraView.disableView();
		}*/
	}
	
	@Override
	public void onResume() {
		
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
	
		Log.i(TAG, "Resumed!");
	}

	/*public void onStop()
	{
		Log.i(TAG, "Stopped!");
		super.onStop();
		
	     if (mOpenCvCameraView != null)
	         mOpenCvCameraView.disableView();
	        
	     
		 SharedPreferences numbers = getSharedPreferences("Numbers", 0);
	     SharedPreferences.Editor editor = numbers.edit();
	     editor.putInt("imgNum", imgNum);
	     editor.commit();
	}*/
	
	@Override
	public void onDestroy(){
		Log.i(TAG, "Destroyed!");
		releaseCVMats();
		
		super.onDestroy();
		if (mOpenCvCameraView != null) {
			mOpenCvCameraView.disableView();
		}
		 SharedPreferences numbers = getSharedPreferences("Numbers", 0);
	     SharedPreferences.Editor editor = numbers.edit();
	     editor.putInt("imgNum", imgNum);
	     editor.commit();
	     
	}
}
