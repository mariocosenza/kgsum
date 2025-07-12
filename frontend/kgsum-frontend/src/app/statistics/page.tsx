"use client"
import {ReactNode} from "react";
import {Cell, Pie, PieChart, ResponsiveContainer, Bar, BarChart, CartesianGrid, XAxis, YAxis, Legend} from "recharts"
import {ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent} from "@/components/ui/chart"
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card"

const datasetDistributionData = [
    {name: "Cross Domain", value: 701, color: "#87CEEB"},
    {name: "Publications", value: 259, color: "#4682B4"},
    {name: "Life Sciences", value: 230, color: "#20B2AA"},
    {name: "Linguistics", value: 191, color: "#5F9EA0"},
    {name: "Geography", value: 92, color: "#48CAE4"},
    {name: "Media", value: 66, color: "#0077BE"},
    {name: "Social Networking", value: 18, color: "#006A96"},
]

// Updated accuracy data with new values
const accuracyData = [
    {
        classifier: "NB",
        VOC: 0.74,
        CURI: 0.59,
        PURI: 0.63,
        LCN: 0.59,
        LPN: 0.59,
        LAB: 0.35,
        COMMENTS: 0.70,
        TLDS: 0.59
    },
    {
        classifier: "KNN",
        VOC: 0.78,
        CURI: 0.45,
        PURI: 0.53,
        LCN: 0.59,
        LPN: 0.55,
        LAB: 0.30,
        COMMENTS: 0.60,
        TLDS: 0.59
    },
    {
        classifier: "SVM",
        VOC: 0.70,
        CURI: 0.48,
        PURI: 0.63,
        LCN: 0.59,
        LPN: 0.59,
        LAB: 0.30,
        COMMENTS: 0.50,
        TLDS: 0.52
    },
    {
        classifier: "J48",
        VOC: 0.63,
        CURI: 0.45,
        PURI: 0.57,
        LCN: 0.38,
        LPN: 0.59,
        LAB: 0.43,
        COMMENTS: 0.47,
        TLDS: 0.55
    },
    {
        classifier: "MLP",
        VOC: 0.56,
        CURI: 0.53,
        PURI: 0.53,
        LCN: 0.47,
        LPN: 0.54,
        LAB: 0.34,
        COMMENTS: 0.54,
        TLDS: 0.57
    },
    {
        classifier: "DEEP",
        VOC: 0.51,
        CURI: 0.51,
        PURI: 0.49,
        LCN: 0.50,
        LPN: 0.57,
        LAB: 0.40,
        COMMENTS: 0.47,
        TLDS: 0.53
    },
    {
        classifier: "BATCHNORM",
        VOC: 0.59,
        CURI: 0.56,
        PURI: 0.60,
        LCN: 0.53,
        LPN: 0.57,
        LAB: 0.34,
        COMMENTS: 0.53,
        TLDS: 0.62
    },
    {
        classifier: "MISTRAL 7B QLORA",
        VOC: 0.59,
        CURI: 0.45,
        PURI: 0.60,
        LCN: 0.41,
        LPN: 0.48,
        LAB: 0.55,
        COMMENTS: 0.57,
        TLDS: 0.55
    },
    {
        classifier: "GEMINI FLASH 2.0",
        VOC: 0.32,
        CURI: 0.43,
        PURI: 0.40,
        LCN: 0.38,
        LPN: 0.40,
        LAB: 0.23,
        COMMENTS: 0.35,
        TLDS: 0.07
    },
    {
        classifier: "MISTRAL 7B",
        VOC: 0.22,
        CURI: 0.28,
        PURI: 0.43,
        LCN: 0.26,
        LPN: 0.19,
        LAB: 0.12,
        COMMENTS: 0.33,
        TLDS: 0.07
    }
]

// Updated F1 score data with new values
const f1ScoreData = [
    {
        classifier: "NB",
        VOC: 0.72,
        CURI: 0.60,
        PURI: 0.63,
        LCN: 0.57,
        LPN: 0.55,
        LAB: 0.34,
        COMMENTS: 0.68,
        TLDS: 0.61
    },
    {
        classifier: "KNN",
        VOC: 0.77,
        CURI: 0.49,
        PURI: 0.54,
        LCN: 0.56,
        LPN: 0.55,
        LAB: 0.21,
        COMMENTS: 0.59,
        TLDS: 0.52
    },
    {
        classifier: "SVM",
        VOC: 0.71,
        CURI: 0.46,
        PURI: 0.63,
        LCN: 0.56,
        LPN: 0.59,
        LAB: 0.14,
        COMMENTS: 0.55,
        TLDS: 0.54
    },
    {
        classifier: "J48",
        VOC: 0.62,
        CURI: 0.43,
        PURI: 0.55,
        LCN: 0.39,
        LPN: 0.53,
        LAB: 0.32,
        COMMENTS: 0.54,
        TLDS: 0.55
    },
    {
        classifier: "MLP",
        VOC: 0.55,
        CURI: 0.48,
        PURI: 0.48,
        LCN: 0.41,
        LPN: 0.52,
        LAB: 0.25,
        COMMENTS: 0.50,
        TLDS: 0.55
    },
    {
        classifier: "DEEP",
        VOC: 0.47,
        CURI: 0.48,
        PURI: 0.47,
        LCN: 0.45,
        LPN: 0.54,
        LAB: 0.35,
        COMMENTS: 0.42,
        TLDS: 0.51
    },
    {
        classifier: "BATCHNORM",
        VOC: 0.55,
        CURI: 0.51,
        PURI: 0.55,
        LCN: 0.46,
        LPN: 0.51,
        LAB: 0.27,
        COMMENTS: 0.48,
        TLDS: 0.58
    },
    {
        classifier: "MISTRAL 7B QLORA",
        VOC: 0.56,
        CURI: 0.47,
        PURI: 0.58,
        LCN: 0.34,
        LPN: 0.51,
        LAB: 0.48,
        COMMENTS: 0.50,
        TLDS: 0.49
    }
]

// Extended data from the additional tables (unchanged)
const extendedAccuracyData = [
    {
        classifier: "NB",
        VOC: 0.79,
        CURI: 0.67,
        PUR: 0.50,
        LCN: 0.74,
        LPN: 0.53,
        LAB: 0.35,
        COMMENTS: 0.53,
        TLDS: 0.60
    },
    {
        classifier: "KNN",
        VOC: 0.79,
        CURI: 0.63,
        PUR: 0.50,
        LCN: 0.70,
        LPN: 0.60,
        LAB: 0.30,
        COMMENTS: 0.50,
        TLDS: 0.57
    },
    {
        classifier: "SVM",
        VOC: 0.71,
        CURI: 0.70,
        PUR: 0.53,
        LCN: 0.70,
        LPN: 0.57,
        LAB: 0.30,
        COMMENTS: 0.53,
        TLDS: 0.57
    },
    {
        classifier: "J48",
        VOC: 0.63,
        CURI: 0.59,
        PUR: 0.53,
        LCN: 0.59,
        LPN: 0.53,
        LAB: 0.43,
        COMMENTS: 0.57,
        TLDS: 0.47
    },
    {
        classifier: "MISTRAL 7B QLORA",
        VOC: 0.67,
        CURI: 0.50,
        PUR: 0.45,
        LCN: 0.50,
        LPN: 0.48,
        LAB: 0.55,
        COMMENTS: 0.50,
        TLDS: 0.53
    },
    {
        classifier: "DEEP",
        VOC: 0.45,
        CURI: 0.43,
        PUR: 0.43,
        LCN: 0.42,
        LPN: 0.42,
        LAB: 0.29,
        COMMENTS: 0.41,
        TLDS: 0.51
    },
    {
        classifier: "BATCHNORM",
        VOC: 0.49,
        CURI: 0.53,
        PUR: 0.55,
        LCN: 0.42,
        LPN: 0.49,
        LAB: 0.32,
        COMMENTS: 0.46,
        TLDS: 0.50
    },
    {
        classifier: "MLP",
        VOC: 0.47,
        CURI: 0.45,
        PUR: 0.49,
        LCN: 0.41,
        LPN: 0.46,
        LAB: 0.28,
        COMMENTS: 0.46,
        TLDS: 0.57
    },
    {
        classifier: "GEMMA 3 12B",
        VOC: 0.06,
        CURI: 0.42,
        PUR: 0.34,
        LCN: 0.29,
        LPN: 0.33,
        LAB: 0.12,
        COMMENTS: 0.17,
        TLDS: 0.06
    },
    {
        classifier: "GEMINI FLASH 2.0",
        VOC: 0.29,
        CURI: 0.38,
        PUR: 0.42,
        LCN: 0.40,
        LPN: 0.36,
        LAB: 0.23,
        COMMENTS: 0.24,
        TLDS: 0.07
    },
    {
        classifier: "MISTRAL 7B",
        VOC: 0.22,
        CURI: 0.28,
        PUR: 0.34,
        LCN: 0.26,
        LPN: 0.22,
        LAB: 0.12,
        COMMENTS: 0.32,
        TLDS: 0.08
    }
]

const extendedF1ScoreData = [
    {
        classifier: "NB",
        VOC: 0.79,
        CURI: 0.69,
        PUR: 0.54,
        LCN: 0.74,
        LPN: 0.55,
        LAB: 0.34,
        COMMENTS: 0.46,
        TLDS: 0.55
    },
    {
        classifier: "KNN",
        VOC: 0.79,
        CURI: 0.70,
        PUR: 0.47,
        LCN: 0.67,
        LPN: 0.63,
        LAB: 0.21,
        COMMENTS: 0.49,
        TLDS: 0.56
    },
    {
        classifier: "SVM",
        VOC: 0.69,
        CURI: 0.68,
        PUR: 0.53,
        LCN: 0.67,
        LPN: 0.57,
        LAB: 0.14,
        COMMENTS: 0.48,
        TLDS: 0.55
    },
    {
        classifier: "J48",
        VOC: 0.64,
        CURI: 0.60,
        PUR: 0.56,
        LCN: 0.56,
        LPN: 0.55,
        LAB: 0.32,
        COMMENTS: 0.56,
        TLDS: 0.52
    },
    {
        classifier: "MISTRAL 7B QLORA",
        VOC: 0.64,
        CURI: 0.48,
        PUR: 0.45,
        LCN: 0.44,
        LPN: 0.49,
        LAB: 0.48,
        COMMENTS: 0.47,
        TLDS: 0.52
    },
    {
        classifier: "DEEP",
        VOC: 0.42,
        CURI: 0.41,
        PUR: 0.44,
        LCN: 0.43,
        LPN: 0.44,
        LAB: 0.20,
        COMMENTS: 0.40,
        TLDS: 0.52
    },
    {
        classifier: "BATCHNORM",
        VOC: 0.43,
        CURI: 0.47,
        PUR: 0.53,
        LCN: 0.45,
        LPN: 0.48,
        LAB: 0.22,
        COMMENTS: 0.44,
        TLDS: 0.50
    },
    {
        classifier: "MLP",
        VOC: 0.44,
        CURI: 0.41,
        PUR: 0.49,
        LCN: 0.43,
        LPN: 0.46,
        LAB: 0.19,
        COMMENTS: 0.46,
        TLDS: 0.57
    }
]

const pieChartConfig = {
    "Cross Domain": {
        label: "Cross Domain",
        color: "#87CEEB",
    },
    "Publications": {
        label: "Publications", 
        color: "#4682B4",
    },
    "Life Sciences": {
        label: "Life Sciences",
        color: "#20B2AA",
    },
    "Linguistics": {
        label: "Linguistics",
        color: "#5F9EA0",
    },
    "Geography": {
        label: "Geography",
        color: "#48CAE4",
    },
    "Media": {
        label: "Media",
        color: "#0077BE",
    },
    "Social Networking": {
        label: "Social Networking",
        color: "#006A96",
    },
} satisfies ChartConfig

const metricsChartConfig = {
    VOC: {
        label: "VOC",
        color: "#4A90E2",
    },
    CURI: {
        label: "CURI",
        color: "#5F9EA0",
    },
    PURI: {
        label: "PURI",
        color: "#20B2AA",
    },
    LCN: {
        label: "LCN",
        color: "#48CAE4",
    },
    LPN: {
        label: "LPN",
        color: "#87CEEB",
    },
    LAB: {
        label: "LAB",
        color: "#0077BE",
    },
    COMMENTS: {
        label: "COMMENTS",
        color: "#006A96",
    },
    TLDS: {
        label: "TLDS",
        color: "#4682B4",
    },
} satisfies ChartConfig

const extendedMetricsChartConfig = {
    VOC: {
        label: "VOC",
        color: "#4A90E2",
    },
    CURI: {
        label: "CURI",
        color: "#5F9EA0",
    },
    PUR: {
        label: "PUR",
        color: "#20B2AA",
    },
    LCN: {
        label: "LCN",
        color: "#48CAE4",
    },
    LPN: {
        label: "LPN",
        color: "#87CEEB",
    },
    LAB: {
        label: "LAB",
        color: "#0077BE",
    },
    COMMENTS: {
        label: "COMMENTS",
        color: "#006A96",
    },
    TLDS: {
        label: "TLDS",
        color: "#4682B4",
    },
} satisfies ChartConfig

const RADIAN = Math.PI / 180;
const renderCustomizedLabel = (props: {
    cx?: number;
    cy?: number;
    midAngle?: number;
    innerRadius?: number;
    outerRadius?: number;
    value?: number;
}) => {
    const { cx = 0, cy = 0, midAngle = 0, innerRadius = 0, outerRadius = 0, value = 0 } = props;
    
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
        <text 
            x={x} 
            y={y} 
            fill="white" 
            textAnchor={x > cx ? 'start' : 'end'} 
            dominantBaseline="central"
            fontSize={12}
            fontWeight="bold"
        >
            {value}
        </text>
    );
};

export default function Statistiche(): ReactNode {
    return (
        <div className="min-h-screen">
            <div className="max-w-7xl mx-auto space-y-8">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-slate-800 mt-1 mb-2">Statistics Dashboard</h1>
                    <p className="text-slate-600">Brief analysis of classifiers performance metrics</p>
                </div>

                {/* First Row - Pie Chart */}
                <div className="flex justify-center">
                    <Card className="w-full max-w-4xl shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                        <CardHeader className="text-center">
                            <CardTitle className="text-2xl text-slate-800">Dataset Distribution</CardTitle>
                            <CardDescription className="text-slate-600">
                                Class distribution of the dataset with Zenodo, LOD Cloud e GitHub
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={pieChartConfig} className="h-[450px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={datasetDistributionData}
                                            cx="50%"
                                            cy="50%"
                                            labelLine={false}
                                            label={renderCustomizedLabel}
                                            outerRadius={140}
                                            fill="#8884d8"
                                            dataKey="value"
                                        >
                                            {datasetDistributionData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <ChartTooltip 
                                            content={<ChartTooltipContent />}
                                            formatter={(value: number | string, name: string) => [
                                                `${value} datasets`,
                                                name
                                            ]}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </ChartContainer>
                            
                            {/* Legend */}
                            <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
                                {datasetDistributionData.map((item, index) => (
                                    <div key={index} className="flex items-center gap-3">
                                        <div 
                                            className="w-4 h-4 rounded-full shadow-sm" 
                                            style={{backgroundColor: item.color}}
                                        />
                                        <span className="text-slate-700 font-medium">{item.name}</span>
                                        <span className="text-slate-500">({item.value})</span>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Second Row - Two Charts Side by Side */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="text-xl text-slate-800">Accuracy No Filter</CardTitle>
                            <CardDescription className="text-slate-600">
                                Accuracy results on Zenodo and LOD Cloud without filtering popular class
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={metricsChartConfig} className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={accuracyData}
                                        margin={{
                                            top: 20,
                                            right: 30,
                                            left: 20,
                                            bottom: 80,
                                        }}
                                        barSize={12}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis 
                                            dataKey="classifier" 
                                            angle={-45}
                                            textAnchor="end"
                                            height={100}
                                            fontSize={9}
                                            stroke="#64748b"
                                        />
                                        <YAxis 
                                            domain={[0, 1]}
                                            tickFormatter={(value) => value.toFixed(1)}
                                            stroke="#64748b"
                                        />
                                        <ChartTooltip 
                                            content={<ChartTooltipContent />}
                                            formatter={(value: number | string, name: string) => [
                                                `${(value as number).toFixed(3)}`,
                                                name
                                            ]}
                                        />
                                        <Legend />
                                        
                                        <Bar dataKey="VOC" fill="var(--color-VOC)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="CURI" fill="var(--color-CURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="PURI" fill="var(--color-PURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LCN" fill="var(--color-LCN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LPN" fill="var(--color-LPN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LAB" fill="var(--color-LAB)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="COMMENTS" fill="var(--color-COMMENTS)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="TLDS" fill="var(--color-TLD)" radius={[2, 2, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartContainer>
                        </CardContent>
                    </Card>

                    <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="text-xl text-slate-800">F1 Score No filter</CardTitle>
                            <CardDescription className="text-slate-600">
                                F1 Score results on Zenodo and LOD Cloud without filtering popular class
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={metricsChartConfig} className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={f1ScoreData}
                                        margin={{
                                            top: 20,
                                            right: 30,
                                            left: 20,
                                            bottom: 80,
                                        }}
                                        barSize={12}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis 
                                            dataKey="classifier" 
                                            angle={-45}
                                            textAnchor="end"
                                            height={100}
                                            fontSize={9}
                                            stroke="#64748b"
                                        />
                                        <YAxis 
                                            domain={[0, 1]}
                                            tickFormatter={(value) => value.toFixed(1)}
                                            stroke="#64748b"
                                        />
                                        <ChartTooltip 
                                            content={<ChartTooltipContent />}
                                            formatter={(value: number | string, name: string) => [
                                                `${(value as number).toFixed(3)}`,
                                                name
                                            ]}
                                        />
                                        <Legend />
                                        
                                        <Bar dataKey="VOC" fill="var(--color-VOC)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="CURI" fill="var(--color-CURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="PURI" fill="var(--color-PURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LCN" fill="var(--color-LCN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LPN" fill="var(--color-LPN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LAB" fill="var(--color-LAB)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="COMMENTS" fill="var(--color-COMMENTS)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="TLDS" fill="var(--color-TLD)" radius={[2, 2, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartContainer>
                        </CardContent>
                    </Card>
                </div>

                {/* Third Row - Extended Charts */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="text-xl text-slate-800">Accuracy With Filter</CardTitle>
                            <CardDescription className="text-slate-600">
                                Accuracy results on Zenodo and LOD Cloud with filter for popular class
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={extendedMetricsChartConfig} className="h-[450px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={extendedAccuracyData}
                                        margin={{
                                            top: 20,
                                            right: 30,
                                            left: 20,
                                            bottom: 100,
                                        }}
                                        barSize={10}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis 
                                            dataKey="classifier" 
                                            angle={-45}
                                            textAnchor="end"
                                            height={120}
                                            fontSize={8}
                                            stroke="#64748b"
                                        />
                                        <YAxis 
                                            domain={[0, 1]}
                                            tickFormatter={(value) => value.toFixed(1)}
                                            stroke="#64748b"
                                        />
                                        <ChartTooltip 
                                            content={<ChartTooltipContent />}
                                            formatter={(value: number | string, name: string) => [
                                                `${(value as number).toFixed(3)}`,
                                                name
                                            ]}
                                        />
                                        <Legend />
                                        
                                        <Bar dataKey="VOC" fill="var(--color-VOC)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="CURI" fill="var(--color-CURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="PUR" fill="var(--color-PUR)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LCN" fill="var(--color-LCN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LPN" fill="var(--color-LPN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LAB" fill="var(--color-LAB)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="COMMENTS" fill="var(--color-COMMENTS)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="TLDS" fill="var(--color-TLD)" radius={[2, 2, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartContainer>
                        </CardContent>
                    </Card>

                    <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="text-xl text-slate-800">Extended F1 Score With Filter</CardTitle>
                            <CardDescription className="text-slate-600">
                                F1 Score results on Zenodo and LOD Cloud without filtering popular class
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ChartContainer config={extendedMetricsChartConfig} className="h-[450px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart
                                        data={extendedF1ScoreData}
                                        margin={{
                                            top: 20,
                                            right: 30,
                                            left: 20,
                                            bottom: 100,
                                        }}
                                        barSize={10}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis 
                                            dataKey="classifier" 
                                            angle={-45}
                                            textAnchor="end"
                                            height={120}
                                            fontSize={8}
                                            stroke="#64748b"
                                        />
                                        <YAxis 
                                            domain={[0, 1]}
                                            tickFormatter={(value) => value.toFixed(1)}
                                            stroke="#64748b"
                                        />
                                        <ChartTooltip 
                                            content={<ChartTooltipContent />}
                                            formatter={(value: number | string, name: string) => [
                                                `${(value as number).toFixed(3)}`,
                                                name
                                            ]}
                                        />
                                        <Legend />
                                        
                                        <Bar dataKey="VOC" fill="var(--color-VOC)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="CURI" fill="var(--color-CURI)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="PUR" fill="var(--color-PUR)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LCN" fill="var(--color-LCN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LPN" fill="var(--color-LPN)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="LAB" fill="var(--color-LAB)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="COMMENTS" fill="var(--color-COMMENTS)" radius={[2, 2, 0, 0]} />
                                        <Bar dataKey="TLDS" fill="var(--color-TLD)" radius={[2, 2, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartContainer>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    )
}