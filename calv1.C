



// ----------- Bins settings ----------
const int Na = 4;   // number of pTa bins
const int Nb = 7;  // number of pTb bins

double ptb0[Nb+1] = {0.0,0.3,0.6,1.0,1.4,1.8,2.2,2.6};//z
double pta0[Na+1] = {0.4,0.8,1.2,1.8,2.6}; //Y
// -----------------------------------

double v11[Na][Nb] = {0.0};
double errv[Na][Nb] = {0.0};

double ptbcenter[Nb] = {0.0};

double ptacenter[Na] = {0.0};

void calculatev11();
void doFit();

void calv1(){
    calculatev11();
    doFit();
}

void calculatev11() {
  // Open input file
  TFile *fin = TFile::Open("200_ampt_chg.root","READ");
  if (!fin || fin->IsZombie()) { cout<<"File not found\n"; return; }
  
  TH3F *hsame = (TH3F*) fin->Get("Same_8");

  TH3F *hsame7 = (TH3F*) fin->Get("Same_7");

  hsame->Add(hsame7,1);
  
  
  //Reference Mean
  for(int i= 0; i<Na; i++){
    TH1D*proja = hsame->ProjectionY();
    proja->GetXaxis()->SetRangeUser(pta0[i], pta0[i+1]);
    ptacenter[i] = proja->GetMean();
    cout<<"Pta0["<<i<<"]"<<"  "<<ptacenter[i]<<endl;
  }

  cout<<endl;
  
  
  //target mean
  for(int i= 0; i<Nb; i++){
    TH1D*projb = hsame->ProjectionZ();
    projb->GetXaxis()->SetRangeUser(ptb0[i], ptb0[i+1]);
    //double binv = projb->GetXaxis()->GetBinWidth(1);
    //cout<<"Hereeeeeeeeeee"<< binv<<endl;
    ptbcenter[i] = projb->GetMean();
    cout<<"Ptb0["<<i<<"]"<<"  "<<ptbcenter[i]<<endl;
  }
  
  
  // Loop over reference particle pT (y axis in your histogram)
  for (int ia=0; ia<Na; ia++) {
    double y1 = hsame->GetYaxis()->FindBin(pta0[ia]+1e-6);
    double y2 = hsame->GetYaxis()->FindBin(pta0[ia+1]-1e-6);
    
    // Loop over target particle pT (z axis)
    for (int ib=0; ib<Nb; ib++) {
      double z1 = hsame->GetZaxis()->FindBin(ptb0[ib]+1e-6);
      double z2 = hsame->GetZaxis()->FindBin(ptb0[ib+1]-1e-6);
      
      // Set ranges and project
      hsame->GetYaxis()->SetRange(y1,y2);
      hsame->GetZaxis()->SetRange(z1,z2);
      
      TH1D *h_same = hsame->ProjectionX("h_same", y1, y2, z1, z2);

      
      double num=0, den=0;
      double numVar=0, denVar=0;
      
      double entriesSame = h_same->GetEntries();
      
      for (int ibin=1; ibin<=h_same->GetNbinsX(); ibin++) {
	    double phi = h_same->GetXaxis()->GetBinCenter(ibin);
	
	    double s  = h_same->GetBinContent(ibin) / entriesSame;
    	double se = h_same->GetBinError(ibin)/entriesSame;
	
	
	    double w  = s;
	    double we = se; //s sqrt(pow(se/m,2) + pow(s*me/(m*m),2)); // error on w
	
	    num    += w * cos(phi);
	    den    += w;
	    numVar += pow(cos(phi)*we,2);
	    denVar += pow(we,2);
      }
      if (den<0) continue;
      	 v11[ia][ib] = num/den;
	    double sigmaNum = sqrt(numVar);
	    double sigmaDen = sqrt(denVar);
	    errv[ia][ib] =  abs((v11[ia][ib])*sqrt(pow(sigmaNum/num,2) + pow(sigmaDen/den,2)));
        //errv[ia][ib] = sqrt(pow(sigmaNum/den,2) + pow(num*sigmaDen/(den*den),2)); 
      
      delete h_same;

      //To print v11 values for each reference and bins
      cout<<"ia="<<ia<<" ib="<<ib<<" V11="<<v11[ia][ib]<<" +- "<<errv[ia][ib]<<endl;
    }
  }
  
  fin->Close();
}

//End of Calculation of v11

/*
  I have to plot v1b vs ptb.
  the formula v11 = v1a. v1b - k . pta .ptb
  
  so I am using parameter for v1b and calculating v1a in a recursive way:
  model->        v1a = v11 + k.pta.ptb/ v1b
  
  taking Nb number of parameters for v1b, I am setting up or changing v1a for one reference pt bin.
  
  ptacenters are for reference bins and ptbcenters are for target pt bin.
  
  so I setup 3 functions, computev1a, model and chi2.
  
*/


std::vector<double> chi2_history;

//Model
double ModelV11(double v1a, double v1b, double pta, double ptb, double K) {
  return v1a * v1b - K * pta * ptb;
}
//Recursive Method to find V1b using V1a
void ComputeV1a(double* v1a, const double* v1b, double K) {
  for (int i = 0; i < Na; i++) {
    double num = 0.0, den = 0.0;
    for (int j = 0; j < Nb; j++) {
      if (errv[i][j] <= 0) continue;
      
      double ratio = (v11[i][j] + K * ptacenter[i] * ptbcenter[j]) / v1b[j];
      double w = 1.0/ (errv[i][j] * errv[i][j]);
      
      num += w * ratio;
      den += w;
    }
    v1a[i] = (den > 0) ? num / den : 0.0;
  }
}

// chi2 function
double ComputeChi2(const double* v1a, const double* v1b, double K) {
  double chi2=0.0;
  for (int i = 0; i < Na; i++) {
    for (int j = 0; j < Nb; j++) {      
      if (errv[i][j] <= 0) continue;
      double model = ModelV11(v1a[i], v1b[j], ptacenter[i], ptbcenter[j], K);
      double diff  = v11[i][j] - model;
      chi2 += pow((diff / errv[i][j]),2);      
    }
  }
  return chi2;
}

// ------------------ MINUIT FCN ------------------
void fcn(Int_t& npar, Double_t* grad, Double_t& f, Double_t* par, Int_t iflag) {
  double v1b[Nb];
  for (int j = 0; j < Nb; j++) v1b[j] = par[j];
  double K = par[Nb];
  
  double v1a[Na];
  ComputeV1a(v1a, v1b, K);
  
  f = ComputeChi2(v1a, v1b, K);
  chi2_history.push_back(f);
}

// ------------------ MAIN FIT ------------------
void doFit() {
  
  //==============================================================//
  //              MINUIT FIT: v1b parameters + K constant          //
  //==============================================================//
  
  // Create Minuit instance with (Nb + 1) parameters
  TMinuit *gMinuit = new TMinuit(Nb + 1);
  gMinuit->SetFCN(fcn);   // Set user-defined chi-square (fcn) function
   
  //==============================================================//
  //               Define fit parameters and limits                //
  //==============================================================//
  //define parameters: v1b[j] + K

  // define parameters: v1b[j] + K
  for (int j = 0; j < Nb; j++) {
    gMinuit->DefineParameter(j, Form("v1b[%d]", j), -0.03, 1e-3, -0.1, 0.2);;
  }
  gMinuit->DefineParameter(Nb, "K", 0.0002, 1e-4, -1, 1);

 
  //==============================================================//
  //                         Run MIGRAD                            //
  //==============================================================//
  double arglist[2];
  arglist[0] = 30000; //calls
  arglist[1] = 1.0;
  int ierflg =0;
  gMinuit->mnexcm("MIGRAD", arglist, 2, ierflg);
  

  //==============================================================//
  //                   Extract fitted parameters                   //
  //==============================================================//
  double par[Nb + 1], v1er[Nb + 1];
  
  for (int j = 0; j <= Nb; j++) {
    gMinuit->GetParameter(j, par[j], v1er[j]);
  }
  
  // Separate parameter results
  double v1b_best[Nb], v1b_errv[Nb];
  for (int j = 0; j < Nb; j++) {
    v1b_best[j] = par[j];
    v1b_errv[j]  = v1er[j];
  }
  
  double K_best = par[Nb];
  double K_best_err = v1er[Nb];
  //==============================================================//
  //                 Compute derived quantities                    //
  //==============================================================//
  double v1a_best[Na];
  ComputeV1a(v1a_best, v1b_best, K_best);
  
  int ndata = Na * Nb;
  int npar  = Nb + 1;
  int ndf   = ndata - npar;
  double chi2 = ComputeChi2(v1a_best, v1b_best, K_best);
  
  //==============================================================//
  //                    Print final results                        //
  //==============================================================//
  cout << "############################################################" << endl;
  cout << "###                 Final Fit Results                   ###" << endl;
  cout << "############################################################" << endl;
  
  cout << "Chi2   = " << chi2  << endl;
  cout << "ndf    = " << ndf   << endl;
  cout << "Chi2/ndf = " << chi2 / ndf << endl;
  cout << "----------------------------------------" << endl;

  Double_t amin, edm, errdef;
  Int_t nvpar, nparx, icstat;
  
  gMinuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
  
  std::cout << "Chi2 = " << amin << std::endl;
  std::cout << "EDM  = " << edm << std::endl;
  std::cout << "ErrDef = " << errdef << std::endl;
  std::cout << "Status = " << icstat << std::endl;
  
  
  for (int j = 0; j < Nb; j++) {
    cout << Form("v1b[%d] = % .6f ± % .6f", j, v1b_best[j], v1b_errv[j]) << endl;
  }
  cout<<"K"<<" =>  "<<K_best <<"+-"<<"  "<<K_best_err<<endl;
  cout << "############################################################" << endl;
  
  
  // ---------------- PLOT v1evena (ptb) ----------------
  TFile*fo = new TFile("v1_chg_200.root","RECREATE");

  TLine *line = new TLine(0.1, 0.0,2.56, 0.0); // from (xMin,0) to (xMax,0)
  line->SetLineStyle(2);   // dashed
  line->SetLineColor(kBlack);
  line->SetLineWidth(1);

  TLegend *l1 = new TLegend(0.10,0.8,0.62,0.9);
  l1->AddEntry((TObject*)0, "(a) 0.4 < p_{T}^{a} < 0.8(GeV/c)","");
  TLegend *l2 = new TLegend(0.05,0.8,0.62,0.9);
  l2->AddEntry((TObject*)0, "(b) 0.8 < p_{T}^{a} < 1.2(GeV/c)","");
  TLegend *l3 = new TLegend(0.05,0.8,0.62,0.9);
  l3->AddEntry((TObject*)0, "(c) 1.2 < p_{T}^{a} < 1.8(GeV/c)","");
  TLegend *l4 = new TLegend(0.05,0.8,0.62,0.9);
  l4->AddEntry((TObject*)0,"(d) 1.8 < p_{T}^{a} < 2.6(GeV/c)","");
  TLegend *l5 = new TLegend(0.30,0.75,0.62,0.87);
  l5->AddEntry((TObject*)0, "(e) 2.6 < p_{T}^{a} < 3.5(GeV/c)","");
  TLegend *le[5] = {l1,l2,l3,l4,l5};
  for(int l=0; l<5; l++){
    le[l]->SetTextSize(0.06);
    le[l]->SetTextFont(42);
    le[l]->SetBorderSize(0);
    le[l]->SetFillStyle(0);
    le[l]->SetTextAlign(22);
    le[l]->SetTextColor(1);
  } 
  
  l1->SetTextSize(0.05);
 

   TLegend*lev = new TLegend(0.12,0.23,0.69,0.43);
  lev->AddEntry((TObject*)0, "0-10%", "");
  lev->AddEntry((TObject*)0, "Au+Au(AMPT-SM)", "");
  lev->AddEntry((TObject*)0, "200GeV", "");
  lev->SetTextAlign(22);
  lev->SetTextFont(42);
  lev->SetTextSize(0.06);
  lev->SetBorderSize(0);
  lev->SetFillStyle(0);
  lev->SetTextColor(1);


  TLegend*lev2 = new TLegend(0.20, 0.60, 0.60, 0.85);
  lev2->AddEntry((TObject*)0, "0-10%", "");
  lev2->AddEntry((TObject*)0, "Au+Au", "");
  lev2->AddEntry((TObject*)0, "200GeV", "");
  lev2->SetTextFont(62);
  lev2->SetTextSize(0.05);
  lev2->SetBorderSize(1);
  lev2->SetFillStyle(0);
   lev2->SetTextColor(1);

 
  TCanvas* c2 = new TCanvas("c2","v1even(pt)",800,600);

  
  TGraphErrors* gr_b = new TGraphErrors(Nb);
  for (int j = 0; j < Nb; j++) {
    gr_b->SetName("V1_Even"); 
    gr_b->SetPoint(j, ptbcenter[j], v1b_best[j]);
    gr_b->SetPointError(j, 0.0, v1b_errv[j]);
    
  }
  gr_b->GetYaxis()->SetTitleSize(0.05);
  gr_b->GetYaxis()->SetTitleOffset(0.9);
  gr_b->SetMarkerStyle(30);
  gr_b->SetMarkerColor(kBlack);
  gr_b->SetMarkerSize(1.5);
  gr_b->SetLineColor(kBlack);
  gr_b->SetTitle("V1_{Even}");
  gr_b->Draw("AP");
  line->Draw("SAME");
  lev2->Draw("SAME");
  
  gr_b->GetXaxis()->SetTitle("p_{T}^{b} (GeV/c)");
  gr_b->GetYaxis()->SetTitle("v_{1}^{even}");
  c2->SaveAs("v1even_fit_200.pdf");
  gr_b->Write();

  
  
  //fit
  
  TCanvas* c1 = new TCanvas("c1","Fit to v11",1200,400);

  double L=0.14, R=0.04, B=0.2, Tm=0.07;
//  c1->SetLeftMargin(L);
 // c1->SetRightMargin(R);
  //c1->SetTopMargin(Tm);
  //c1->SetBottomMargin(B);
 

   c1->Divide(Na,1,0,0); 
  // Main plotting area
	  

 // gPad->SetTickx(1);
 // gPad->SetTicky(1);
 
  c1->cd(1);
  gPad->SetLeftMargin(L);

  // All pads → same top & bottom margins
  for (int p = 1; p <= 4; ++p) {
      c1->cd(p);
      gPad->SetBottomMargin(B);
      gPad->SetTopMargin(Tm);
  }

  // Pads 2–4 → remove left margin (shared Y-axis style)
  for (int p = 2; p <= 4; ++p) {
      c1->cd(p);
      gPad->SetLeftMargin(0);
  }

  // Last pad → extra right margin (for legend/space)
  c1->cd(4);
  gPad->SetRightMargin(R);
  
  
  
  
  
  
  for (int i = 0; i < Na; i++) {
    c1->cd(i+1);
    //gPad->Clear();
    //gPad->DrawFrame(0, -0.005, 4,0.008);
    TGraphErrors* gr = new TGraphErrors(Nb);

    for (int j = 0; j < Nb; j++) {
      gr->SetName(Form("gr_v11_%d", i+1)); 
      gr->SetPoint(j, ptbcenter[j], v11[i][j]);
      gr->SetPointError(j, 0.1, errv[i][j]);
    }
    
    
    
    
    gr->GetYaxis()->SetMaxDigits(3);
    gr->GetYaxis()->SetRangeUser(-0.0015,0.0015);
    gr->GetYaxis()->SetNdivisions(505);
    //gr->GetYaxis()->SetNoExponent(kFALSE);
    gr->Draw("APE");
    

    gr->SetMarkerStyle(24);
    gr->SetMarkerColor(kRed);
    gr->SetLineColor(kRed);
    gr->SetMarkerSize(1.4);
    
    // Titles only on outer axes
    gr->SetTitle("");
   
    
 


   

    

    
    // X title only on last pad
    if (i == 2) {
      gr->GetXaxis()->SetTitle("");
      gr->GetXaxis()->SetTitleSize(0.07);
      gr->GetXaxis()->CenterTitle();
      gr->GetXaxis()->SetTitleFont(62);
    }
   

    if(i==0)gr->GetYaxis()->SetTitle("v_{11}");	gr->GetYaxis()->CenterTitle();gr->GetYaxis()->SetTitleFont(62);gr->GetYaxis()->SetTitleSize(0.07);gr->GetYaxis()->SetTitleOffset(0.9);
    if(i==0)gr->GetYaxis()->SetLabelOffset(0.02);
    gr->GetXaxis()->SetLabelSize(0.06);
    gr->GetXaxis()->SetLabelFont(62);
    gr->GetYaxis()->SetLabelSize(0.06);
    gr->GetYaxis()->SetLabelFont(62);
    
    
    
    // allow exponent
    // Y title only on first pad
    if (i == 0) {
    } else {
      gr->GetYaxis()->SetTitle("");
      gr->GetYaxis()->SetLabelSize(0);
    }
    //gPad->Update();
    line->Draw("SAME");
    if(i==0){
      lev->Draw("SAME");
      
    }


    double xmin = 0.0;
    double xmax = 2.6;
    double eps  = 0.05;
    gr->GetXaxis()->SetLimits(xmin - eps, xmax + eps);
    if(i==0){
      
      l1->Draw("SAME");
    }
    if(i==1){
      
      l2->Draw("SAME");
    }
    if(i==2){
	    
      l3->Draw("SAME");
    }
    if(i==3){
	    
      l4->Draw("SAME");
    }
    if(i==4){
	   
      l5->Draw("SAME");
    }

    
    TGraph* gr_fit = new TGraph(Nb);
    for (int j = 0; j < Nb; j++) {
      double model = ModelV11(v1a_best[i], v1b_best[j], ptacenter[i], ptbcenter[j], K_best);
      gr_fit->SetPoint(j, ptbcenter[j], model);
    }
    gr_fit->SetTitle(" ");
    gr_fit->SetLineColor(4);
    gr_fit->SetLineWidth(2);
    gr_fit->SetLineStyle(1);
    gr_fit->Draw("C SAME");
    gr->Write();
    
    
    TLegend*lfit = new TLegend(0.2,0.65,0.76,0.77);
    lfit->AddEntry(gr,"Ch. Hadrons","PE");
    lfit->AddEntry(gr_fit, "fit","L");
    lfit->SetBorderSize(0);
    lfit->SetFillStyle(0);
    lfit->SetTextSize(0.05);
    lfit->SetNColumns(2);
    
    if(i==0)lfit->Draw("same");
    
    //c1->cd();  // go to full canvas
    
    if(i==0){
	    gr->GetXaxis()->SetLabelSize(0.0558);
    }




  }
  
  c1->cd(0);
TPad* labelPad = new TPad("labelPad","",0,0,1,1);
labelPad->SetFillStyle(4000);   // fully transparent
labelPad->SetFillColor(0);
labelPad->SetBorderSize(0);
labelPad->Draw();
labelPad->cd();

TLatex* xLabel = new TLatex();
xLabel->SetNDC();
xLabel->SetTextFont(62);
xLabel->SetTextSize(0.056);
xLabel->SetTextAlign(22);
xLabel->DrawLatex(0.5, 0.06, "p_{T}^{b} (GeV/c)");

c1->Modified();
c1->Update();

c1->SaveAs("v11_200_chg.pdf");
  
  // Print summary
  for (int i = 0; i < Na; i++) {
    cout << Form("v1a(pt=%.3f) = % .6e", ptacenter[i], v1a_best[i]) << endl;
  }
  
  cout<<"//***************************************//"<<endl;
  
  cout<<"The V1even We need to Plot"<<endl;
  
  cout<<"K"<<" =>  "<<K_best<<endl;
  
  cout<<"//******//"<<endl;
  for (int j = 0; j < Nb; j++) {
    cout << Form("v1b(pt=%.3f) = % .6e", ptbcenter[j], v1b_best[j]) << endl;
  }
  
  cout<<"//********************************************//"<<endl;
  

  

  fo->Close();
}



//End of analysis.
