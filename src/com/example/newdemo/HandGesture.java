package com.example.newdemo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class HandGesture {
	public List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	public int cMaxId = -1;
	public Mat hie = new Mat();
	public List<MatOfPoint> hullP = new ArrayList<MatOfPoint>();
	public MatOfInt hullI = new MatOfInt();
	public Rect boundingRect;
	public MatOfInt4 defects = new MatOfInt4();
	
	public ArrayList<Integer> defectIdAfter = new ArrayList<Integer>();
	
	//public MatOfPoint fingerTips = new MatOfPoint();
	public List<Point> fingerTips = new ArrayList<Point>();
	public List<Point> fingerTipsOrder = new ArrayList<Point>();
	public Map<Double, Point> fingerTipsOrdered = new TreeMap<Double, Point>();
	
	public MatOfPoint2f defectMat = new MatOfPoint2f();
	public List<Point> defectPoints = new ArrayList<Point>();
	public Map<Double, Integer> defectPointsOrdered = new TreeMap<Double, Integer>();
	
	public Point palmCenter = new Point();
	public MatOfPoint2f hullCurP = new MatOfPoint2f();
	public MatOfPoint2f approxHull = new MatOfPoint2f();
	
	public MatOfPoint2f approxContour = new MatOfPoint2f();
	
	public MatOfPoint palmDefects = new MatOfPoint();
	
	public Point momentCenter = new Point();
	public double momentTiltAngle;
	
	public Point inCircle = new Point();
//	public double inCircle_x;
	//public double inCircle_y;
	public double inCircleRadius;
	
	public List<Double> features = new ArrayList<Double>();
	
	private boolean isHand = false;
	
	private float[] palmCircleRadius = {0};
	
	
	void findBiggestContour() 
	{
		int idx = -1;
		int cNum = 0;
		
		for (int i = 0; i < contours.size(); i++)
		{
			int curNum = contours.get(i).toList().size();
			if (curNum > cNum) {
				idx = i;
				cNum = curNum;
			}
		}
		
		cMaxId = idx;
	}
	
	boolean detectIsHand(Mat img)
	{
		int centerX = 0;
		int centerY = 0;
		if (boundingRect != null) {
			centerX = boundingRect.x + boundingRect.width/2;
			centerY = boundingRect.y + boundingRect.height/2;
		}
		if (cMaxId == -1)
			isHand = false;
		else if (boundingRect == null) {
			isHand = false;
		} else if ((boundingRect.height == 0) || (boundingRect.width == 0))
			isHand = false;
		else if ((centerX < img.cols()/4) || (centerX > img.cols()*3/4))
			isHand = false;
		else
			isHand = true;
		return isHand;
	}
	
	String feature2SVMString(int label)
	{
		String ret = Integer.toString(label) + " ";
		int i;
		for (i = 0; i < features.size(); i++)
		{
			int id = i + 1;
			ret = ret + id + ":" + features.get(i) + " ";
		}
		ret = ret + "\n";
		return ret;
	}
	
	String featureExtraction(Mat img, int label)
	{
		String ret = null;
		if ((detectIsHand(img))) {
			//ret = Integer.toString(label) + " ";
			
			defectMat.fromList(defectPoints);
			//Imgproc.minEnclosingCircle(defectMat, palmCenter, palmCircleRadius);
			
			
		//	Moments mu = Imgproc.moments(contours.get(cMaxId));
		//	palmCenter.x = mu.get_m10()/mu.get_m00();
		//	palmCenter.y = mu.get_m01()/mu.get_m00();
			
		//	Core.circle(img, palmCenter, 3, new Scalar(240,240,45,0), -5);
			
			
			//Core.circle(img, palmCenter, (int)palmCircleRadius[0], new Scalar(240,240,45,0), 2);
			
			
		/*	fingerTipsOrder.clear();
			List<Integer> dList = defects.toList();
			features.clear();
			
			int count = 0;
			for (Map.Entry entry: defectPointsOrdered.entrySet()) 
			{
				int id = (Integer)entry.getValue();
				double curAngle = (Double)entry.getKey();
				
				Point dPoint = (contours.get(cMaxId).toArray())[dList.get(id)];
				Point fPoint0 = (contours.get(cMaxId).toArray())[dList.get(id-2)];
				Point fPoint1 = (contours.get(cMaxId).toArray())[dList.get(id-1)];
				
				features.add(fPoint0.x-dPoint.x);
				features.add(fPoint0.y-dPoint.y);
				features.add(fPoint1.x-dPoint.x);
				features.add(fPoint1.y-dPoint.y);
				
				Core.circle(img, dPoint, 2, new Scalar(0,0,255,0), -5);
				Core.line(img, dPoint, fPoint0, new Scalar(24, 77, 9), 3);
				Core.line(img, dPoint, fPoint1, new Scalar(128, 24, 201), 3);
				
				Core.putText(img, Integer.toString(count), new Point(dPoint.x - 10, 
						dPoint.y + 10), Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(0));
				
				count++;
				
			}*/
			List<Integer> dList = defects.toList();
			Point[] contourPts = contours.get(cMaxId).toArray();
			Point prevDefectVec = null;
			int i;
			for (i = 0; i < defectIdAfter.size(); i++)
			{
				int curDlistId = defectIdAfter.get(i);
				int curId = dList.get(curDlistId);
				
				Point curDefectPoint = contourPts[curId];
				Point curDefectVec = new Point();
				curDefectVec.x = curDefectPoint.x - inCircle.x;
				curDefectVec.y = curDefectPoint.y - inCircle.y;
				
				if (prevDefectVec != null) {
					double dotProduct = curDefectVec.x*prevDefectVec.x +
							curDefectVec.y*prevDefectVec.y;
					double crossProduct = curDefectVec.x*prevDefectVec.y - 
							prevDefectVec.x*curDefectVec.y;
					
					if (crossProduct <= 0)
						break;
				}
				
				
				prevDefectVec = curDefectVec;
				
			//	Core.circle(img, curDefectPoint, 2, new Scalar(0, 0, 255), -5);
			//	Core.putText(img, Integer.toString(i), new Point(curDefectPoint.x - 10, 
			//			curDefectPoint.y + 10), Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(0));
				
			}
			
			int startId = i;
			int countId = 0;
			
			ArrayList<Point> finTipsTemp = new ArrayList<Point>();
			
			if (defectIdAfter.size() > 0) {
				boolean end = false;
				
				for (int j = startId; ; j++)
				{
					if (j == defectIdAfter.size())
							{
						
						if (end == false) {
							j = 0;
							end = true;
						}
						else 
							break;
					}
					
					
					
					if ((j == startId) && (end == true))
						break;
					
					int curDlistId = defectIdAfter.get(j);
					int curId = dList.get(curDlistId);
					
					Point curDefectPoint = contourPts[curId];
					Point fin0 = contourPts[dList.get(curDlistId-2)];
					Point fin1 = contourPts[dList.get(curDlistId-1)];
					finTipsTemp.add(fin0);
					finTipsTemp.add(fin1);
					
					Core.circle(img, curDefectPoint, 2, new Scalar(0, 0, 255), -5);
				//	Core.putText(img, Integer.toString(countId), new Point(curDefectPoint.x - 10, 
				//			curDefectPoint.y + 10), Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(0));
					
					countId++;
				}
			
			}
			
			int count = 0;
			features.clear();
			for (int fid = 0; fid < finTipsTemp.size(); )
			{
				if (count > 5)
					break;
				
				Point curFinPoint = finTipsTemp.get(fid);
				
				if ((fid%2 == 0)) {
					
				    if (fid != 0) {
				    	Point prevFinPoint = finTipsTemp.get(fid-1);
						curFinPoint.x = (curFinPoint.x + prevFinPoint.x)/2;
						curFinPoint.y = (curFinPoint.y + prevFinPoint.y)/2;
				    }
					
					
					if (fid == (finTipsTemp.size() - 2) )
						fid++;
					else
						fid += 2;
				} else
					fid++;
				
				
				Point disFinger = new Point(curFinPoint.x-inCircle.x, curFinPoint.y-inCircle.y);
				double dis = Math.sqrt(disFinger.x*disFinger.x+disFinger.y*disFinger.y);
				Double f1 = (disFinger.x)/inCircleRadius;
				Double f2 = (disFinger.y)/inCircleRadius;
				features.add(f1);
				features.add(f2);
				
				Core.line(img, inCircle, curFinPoint, new Scalar(24, 77, 9), 2);
				Core.circle(img, curFinPoint, 2, Scalar.all(0), -5);
				
				Core.putText(img, Integer.toString(count), new Point(curFinPoint.x - 10, 
									curFinPoint.y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(0));
				
				count++;
							
			}
			
			ret = feature2SVMString(label);
			
			/*int count = 0;
			int fid = 1;
			features.clear();
			
			Point prev = null;
			double firstAngle = 0;
			for (Iterator<Map.Entry<Double, Point>>it=fingerTipsOrdered.entrySet().iterator();
					it.hasNext();) 
			{
				if (count >= 5)
					break;
				
				Map.Entry<Double, Point> entry = it.next();
				
				Point curTip = (Point)entry.getValue();
				double curAngle = (Double)entry.getKey();
				
				if (prev!=null) {
					Point dis = new Point(curTip.x-prev.x, curTip.y-prev.y);
					if (dis.x*dis.x + dis.y*dis.y < Math.pow(inCircleRadius*0.6, 2)) {
						prev = curTip;
						continue;
					}
					
			
				} else {
					firstAngle = curAngle;
					
				}
				prev = curTip;
				
				Point disFinger = new Point(curTip.x-inCircle.x, curTip.y-inCircle.y);
				double dis = Math.sqrt(disFinger.x*disFinger.x+disFinger.y*disFinger.y);
				if (dis < inCircleRadius*1.5)
					continue;
				
				Double f1 = (disFinger.x)/inCircleRadius;
				Double f2 = (disFinger.y)/inCircleRadius;
				features.add(f1);
				features.add(f2);
				
				if (f1 != 0)
					ret = ret + fid + ":" + f1.toString() + " ";
				fid++;
				
				if (f2!= 0)
					ret = ret + fid + ":" + f2.toString() + " ";
				fid++;
				
				Core.line(img, inCircle, curTip, new Scalar(24, 77, 9), 2);
				Core.circle(img, curTip, 2, Scalar.all(0), -5);
				
			//	Core.putText(img, Integer.toString(count), new Point(curTip.x - 10, 
			//			curTip.y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar.all(0));
				
				
				
				count++;
			}
			
			for (; count < 5; count++) {
				features.add(0.0);
				features.add(0.0);
			}
			
			ret = ret + "\n";*/
		
		}
		
		return ret;
	}
	
	public double calculateTilt(double m11, double m20, double m02)
	{
	  double diff = m20 - m02;
	/*  if (diff == 0) {
	    if (m11 == 0)
	      return 0;
	    else if (m11 > 0)
	      return (Math.PI/4);
	    else   // m11 < 0
	      return -(Math.PI/4);
	  }*/

	  double theta = 0.5 * Math.atan2(2*m11, diff);
	  //int tilt = (int) Math.round( Math.toDegrees(theta));

	/*  if ((diff > 0) && (m11 == 0))
	    return 0;
	  else if ((diff < 0) && (m11 == 0))
	    return -(Math.PI/2);
	  else if ((diff > 0) && (m11 > 0))  // 0 to 45 degrees
	    return theta;
	  else if ((diff > 0) && (m11 < 0))  // -45 to 0
	    return (Math.PI + theta);   // change to counter-clockwise angle
	  else if ((diff < 0) && (m11 > 0))   // 45 to 90
	    return theta;
	  else if ((diff < 0) && (m11 < 0))   // -90 to -45
	    return (Math.PI + theta);  // change to counter-clockwise angle

	  System.out.println("Error in moments for tilt angle");
	  return 0;*/
	  momentTiltAngle = theta;
	  
	  return theta;
	  
	}  // end of calculateTilt()
	
	public native double findInscribedCircleJNI(long imgAddr, double rectTLX, double rectTLY,
			double rectBRX, double rectBRY, double[] incircleX, double[] incircleY, long contourAddr);
	
	void findInscribedCircle(Mat img)
	{
		
		/*int rowNum = img.rows();
		int colNum = img.cols();
		
		if (boundingRect!=null) {
			Point tl = boundingRect.tl();
			Point br = boundingRect.br();
			
			inCircleRadius = 0;
			double targetX = 0;
			double targetY = 0;
			
			for (int y = (int)tl.y; y < (int)br.y; y++)
			{
				for (int x = (int)tl.x; x < (int)br.x; x++)
				{
					double curDist = Imgproc.pointPolygonTest(approxContour, new Point(x, y), true);
					if (curDist > inCircleRadius) {
						inCircleRadius = curDist;
						targetX = x;
						targetY = y;
					}
				}
			}
			
			inCircle.x = targetX;
			inCircle.y = targetY;
			
			Core.circle(img, inCircle, (int)inCircleRadius, new Scalar(240,240,45,0), 2);
			
		}*/
		
		Point tl = boundingRect.tl();
		Point br = boundingRect.br();
		
		double[] cirx = new double[]{0};
		double[] ciry = new double[]{0};
		
		inCircleRadius = findInscribedCircleJNI(img.getNativeObjAddr(), tl.x, tl.y, br.x, br.y, cirx, ciry, 
				approxContour.getNativeObjAddr());
		inCircle.x = cirx[0];
		inCircle.y = ciry[0];
		
		Core.circle(img, inCircle, (int)inCircleRadius, new Scalar(240,240,45,0), 2);
		Core.circle(img, inCircle, 3, Scalar.all(0), -2);
	}
		
		
	
}
