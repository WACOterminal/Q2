from fastapi import APIRouter
from managerQ.app.core.report_generator import report_generator

router = APIRouter()

@router.get("/finops")
async def get_finops_report():
    return await report_generator.generate_finops_summary()

@router.get("/security")
async def get_security_report():
    return await report_generator.generate_security_summary()

@router.get("/rca")
async def get_rca_report():
    return await report_generator.generate_rca_summary()

@router.get("/strategic-briefing")
async def get_strategic_briefing():
    return await report_generator.generate_strategic_briefing()

@router.get("/predictive-scaling")
async def get_predictive_scaling_report():
    return await report_generator.generate_predictive_scaling_summary() 