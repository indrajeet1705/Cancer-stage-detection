import React, { useState } from "react";
import { IoInformationCircleOutline } from "react-icons/io5";
import { IoIosCheckmarkCircleOutline } from "react-icons/io";
import axios from "axios";
import { Atom, LifeLine } from "react-loading-indicators";
const Home = () => {
  const [age,setAge]=useState('')
  const [tumorGrade,setTumorGrade]=useState('')
  const [TStage,setTStage]=useState('')
  const [NStage,setNStage]=useState('')
  const [estrogen,setEstrogen]=useState('')
  const [progesterone,setProgesteron]=useState('')
  const [tumerSize,setTumerSize]=useState('')
  const [predictedStage,setPredictedStage]=useState('')
  const [simpleStage,setSimpleStage]=useState('')
  const handleSubmit=async()=>{
    const inputs={
      'Age':age,
      'Tumor Size':tumerSize,
      'Grade':tumorGrade,
      'T Stage':TStage,
      'N Stage':NStage,
      'Estrogen Status':estrogen,
      'Progesterone Status':progesterone,

    }
    console.log(inputs)
    const response= await axios.post('http://127.0.0.1:5000/predict',inputs)
    console.log(response)
    if( response.data){
      setPredictedStage(response.data.predicted_stage)
      setSimpleStage(response.data.simple_stage)
    }

  }
  return (
    <div className="w-full flex flex-col">
      {/* Heading */}
      <div className="flex gap-5 mx-auto flex-col items-center mt-10">
        <h1 className="text-5xl flex gap-5 font-extrabold bg-gradient-to-r from-blue-600 via-pink-400 to-yellow-400 bg-clip-text text-transparent">
           Cancer Stage Detection
        </h1>
        <p className="text-slate-600 text-xl text-center max-w-2xl">
          Advanced AI-powered breast cancer staging analysis. Please enter the
          required medical parameters below.
        </p>
      </div>

      {/* Form Section */}
      <div className="flex mt-10 justify-evenly">
        <div className="w-[800px] border border-slate-300 shadow-md p-8 rounded-xl bg-white">
          {/* Title */}
          <h1 className="flex gap-2 items-center text-slate-700 text-xl font-semibold">
            <IoInformationCircleOutline size={30} />
            Patient Details
          </h1>
          <p className="text-sm text-slate-600 mt-2">
            Enter all required medical parameters for accurate staging analysis
          </p>
          <div className="flex justify-between">
            <div className="w-[300px]">
              {/* Age */}
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">
                  Age (in years)
                </label>
                <input
                  type="text"
                  value={age}
                  onChange={(e)=>setAge(e.target.value)}
                  placeholder="Enter age"
                  className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400"
                />
              </div>

              {/* Tumor Size */}
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">Tumor Size</label>
                <input
                value={tumerSize}
                onChange={e=>setTumerSize(e.target.value)}
                  type="text"
                  placeholder="Tumor size in mm"
                  className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400"
                />
              </div>

              {/* Tumor Grade */}
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">
                  Tumor Grade
                </label>
                <select 
                onChange={e=>setTumorGrade(e.target.value)}
                value={tumorGrade}
                className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400">
                    <option value="" disabled>Select Grade</option>

                  <option value="Well differentiated; Grade I">Grade I: Well differentiated</option>
                  <option value="Moderately differentiated; Grade II">Grade II: Moderately differentiated</option>
                  <option value="Poorly differentiated; Grade III">Grade III: Poorly differentiated</option>
                  <option value="Undifferentiated; Grade IV">Grade IV: Undifferentiated</option>
                </select>
              </div>
              {/* T Stage */}
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">T Stage</label>
                <select 
                onChange={e=>setTStage(e.target.value)}
                value={TStage}
                className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400">
                    <option value="" disabled>Select T Stage</option>

                  <option value="T1">T1</option>
                  <option value="T2">T2</option>
                  <option value="T3">T3</option>
                  <option value="T4">T4</option>
                </select>
              </div>
            </div>

            <div className=" w-[300px]">
              {/* N Stage */}
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">N Stage</label>
                <select
                value={NStage}
                onChange={e=>setNStage(e.target.value)}
                className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400">
                    <option value="" disabled>Select N Stage</option>

                  
                  <option value="N1">N1</option>
                  <option value="N2">N2</option>
                  <option value="N3">N3</option>
                </select>
              </div>
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">
                  Estrogen Status
                </label>
                <select
                value={estrogen}
                onChange={e=>setEstrogen(e.target.value)}
                className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400">
                    <option value="" disabled>Select Estrogen</option>

                  <option value="Positive">Positive</option>
                  <option value="Negative">Negative</option>
                </select>
              </div>
              <div className="gap-2 flex flex-col mt-4">
                <label className="text-slate-700 font-medium">
                  Progesterone Status
                </label>
                <select
                value={progesterone}
                onChange={e=>setProgesteron(e.target.value)}
                className="rounded-lg w-full outline-none border border-slate-300 p-2 focus:ring-2 focus:ring-blue-400">
                    <option value="" disabled>Select Progesteron</option>

                  <option value="Positive">Positive</option>
                  <option value="Negative">Negative</option>
                </select>
              </div>
              <button onClick={handleSubmit} className=" w-full p-3 justify-center flex mx-auto mt-8 bg-gradient-to-r from-blue-600 font-semibold to-yellow-900  text-lg rounded-2xl text-white">Detect Cancer Stage</button>
            </div>
          </div>
        </div>

        {/* Right Side */}
        <div className="w-[400px] border border-slate-300 shadow-md rounded-xl p-6 bg-white">
        <h1 className=" flex font-semibold text-slate-700 text-xl gap-3 "><IoIosCheckmarkCircleOutline size={30} color="green"/> Result</h1>
         {
          predictedStage ?<div className="  p-2 gap-6 flex flex-col text-xl text-slate-700"> <h2>{predictedStage}
          </h2><h2>{simpleStage}</h2></div>:
          <div className=" w-full h-full items-center justify-center flex"><LifeLine  size="medium" text="" textColor="" /></div>
         }
        </div>
      </div>
    </div>
  );
};

export default Home;
